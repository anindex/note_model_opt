from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RunSpec:
    name: str
    amp_dtype: Optional[str] = None  # "bfloat16" | "float16" | None
    tf32: bool = False
    flash_sdp: bool = False
    mem_efficient_sdp: bool = True
    math_sdp: bool = True


@dataclass
class RunResult:
    name: str
    ok: bool
    returncode: int
    wall_s: float
    episodes_per_s: Optional[float]
    approx_steps_per_s: Optional[float]
    success_rate: Optional[float]
    policy_latency_mean_ms: Optional[float]
    policy_latency_p50_ms: Optional[float]
    policy_latency_p90_ms: Optional[float]
    details_path: str
    stdout_tail: str
    stderr_tail: str
    spec: Dict[str, Any]


def _tail(s: str, n: int = 2000) -> str:
    s = s or ""
    return s[-n:]


def _write_sitecustomize(site_dir: Path) -> None:
    """
    Injects:
      * torch backend knobs (TF32, SDPA preferences)
      * optional AMP autocast wrapper for XVLAPolicy.select_action
      * policy latency logging to ${LEROBOT_OUTPUT_DIR}/policy_latency.json
    """
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "sitecustomize.py").write_text(
        r'''
import atexit
import json
import os
import time
from contextlib import nullcontext

try:
    import torch

    def _env_bool(k: str, default: str = "0") -> bool:
        return os.getenv(k, default).strip() in ("1", "true", "True", "yes", "YES")

    def _env_str(k: str, default: str = "") -> str:
        return os.getenv(k, default).strip()

    # Backend knobs
    if torch.cuda.is_available():
        tf32 = _env_bool("LEROBOT_TF32", "0")
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        if tf32:
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        flash = _env_bool("LEROBOT_FLASH_SDP", "0")
        mem_eff = _env_bool("LEROBOT_MEM_EFF_SDP", "1")
        math = _env_bool("LEROBOT_MATH_SDP", "1")
        try:
            torch.backends.cuda.enable_flash_sdp(flash)
            torch.backends.cuda.enable_mem_efficient_sdp(mem_eff)
            torch.backends.cuda.enable_math_sdp(math)
        except Exception:
            pass

    # AMP wrapper and latency logging
    amp_dtype = _env_str("LEROBOT_AMP_DTYPE", "")
    out_dir = _env_str("LEROBOT_OUTPUT_DIR", "")

    lat_ms = []

    def _percentile(xs, p):
        if not xs:
            return None
        xs = sorted(xs)
        k = (len(xs) - 1) * p
        f = int(k)
        c = min(f + 1, len(xs) - 1)
        if f == c:
            return xs[f]
        return xs[f] * (c - k) + xs[c] * (k - f)

    def _dump_latency():
        if not out_dir:
            return
        if not lat_ms:
            payload = {"n_calls": 0, "mean_ms": None, "p50_ms": None, "p90_ms": None}
        else:
            mean_ms = sum(lat_ms) / len(lat_ms)
            payload = {
                "n_calls": int(len(lat_ms)),
                "mean_ms": float(mean_ms),
                "p50_ms": float(_percentile(lat_ms, 0.50)),
                "p90_ms": float(_percentile(lat_ms, 0.90)),
            }
        try:
            p = os.path.join(out_dir, "policy_latency.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    atexit.register(_dump_latency)

    # Determine AMP dtype if specified
    amp_torch_dtype = None
    if amp_dtype:
        if amp_dtype == "bfloat16":
            amp_torch_dtype = torch.bfloat16
        elif amp_dtype == "float16":
            amp_torch_dtype = torch.float16

    # Always wrap select_action for latency timing, optionally with AMP
    try:
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

        orig = XVLAPolicy.select_action

        def wrapped(self, obs):
            # Use AMP context only if dtype is specified
            if amp_torch_dtype is not None and torch.cuda.is_available():
                ctx = torch.autocast(device_type="cuda", dtype=amp_torch_dtype)
            else:
                ctx = nullcontext()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with ctx:
                out = orig(self, obs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            lat_ms.append((t1 - t0) * 1000.0)
            return out

        XVLAPolicy.select_action = wrapped
    except Exception:
        pass

except Exception:
    pass
'''
    )


def _run_cmd(cmd: List[str], env: Dict[str, str]) -> Tuple[int, str, str, float]:
    t0 = time.perf_counter()
    p = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    t1 = time.perf_counter()
    return p.returncode, p.stdout, p.stderr, (t1 - t0)


def _find_metrics_files(out_dir: Path) -> List[Path]:
    pats = [
        "**/eval_info.json",
        "**/eval_infos.json",
        "**/metrics.json",
        "**/results.json",
        "**/summary.json",
        "**/*eval*.json",
    ]
    out: List[Path] = []
    seen = set()
    for pat in pats:
        for p in out_dir.glob(pat):
            rp = str(p.resolve())
            if rp not in seen and p.is_file():
                seen.add(rp)
                out.append(p)
    return out


def _load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _extract_success_from_any(obj: Any) -> Optional[float]:
    def normalize(v: float) -> Optional[float]:
        if v > 1.0 and v <= 100.0:
            v = v / 100.0
        if 0.0 <= v <= 1.0:
            return v
        return None

    if isinstance(obj, list) and obj:
        vals: List[float] = []
        for ep in obj:
            if not isinstance(ep, dict):
                continue
            for k in ep.keys():
                if "success" in k.lower():
                    v = ep[k]
                    if isinstance(v, bool):
                        vals.append(1.0 if v else 0.0)
                        break
                    if isinstance(v, (int, float)):
                        vals.append(float(v))
                        break
        if vals:
            mx = max(vals)
            if mx > 1.0 and mx <= 100.0:
                vals = [v / 100.0 for v in vals]
            vals = [1.0 if v >= 1.0 else 0.0 if v <= 0.0 else v for v in vals]
            return sum(vals) / len(vals)

        for it in obj:
            r = _extract_success_from_any(it)
            if r is not None:
                return r
        return None

    if isinstance(obj, dict):
        for k, v in obj.items():
            if "success" in str(k).lower():
                if isinstance(v, bool):
                    return 1.0 if v else 0.0
                if isinstance(v, (int, float)):
                    r = normalize(float(v))
                    if r is not None:
                        return r

        for v in obj.values():
            r = _extract_success_from_any(v)
            if r is not None:
                return r

    return None


def _parse_success_rate(out_dir: Path) -> Tuple[Optional[float], List[str]]:
    notes: List[str] = []
    files = _find_metrics_files(out_dir)
    if not files:
        notes.append("No metrics json found under output_dir.")
        return None, notes

    for f in files:
        obj = _load_json(f)
        r = _extract_success_from_any(obj)
        if r is not None:
            notes.append(f"Success rate parsed from: {f}")
            return r, notes

    notes.append("Metrics json found but could not parse success rate.")
    notes.append("Files scanned:\n" + "\n".join(str(p) for p in files[:10]))
    return None, notes


def _parse_policy_latency(out_dir: Path) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Parse policy_latency.json and return (mean_ms, p50_ms, p90_ms)."""
    latency_file = out_dir / "policy_latency.json"
    if not latency_file.exists():
        return None, None, None
    data = _load_json(latency_file)
    if not data or not isinstance(data, dict):
        return None, None, None
    return (
        data.get("mean_ms"),
        data.get("p50_ms"),
        data.get("p90_ms"),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", type=str, default="./bench_xvla_libero")
    ap.add_argument("--policy_path", type=str, default="lerobot/xvla-libero")
    ap.add_argument("--tasks", type=str, default="libero_spatial,libero_goal,libero_10")
    ap.add_argument("--control_mode", type=str, default="absolute")
    ap.add_argument("--episode_length", type=int, default=800)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--n_episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=142)
    ap.add_argument("--max_parallel_tasks", type=int, default=1)
    args = ap.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    site_dir = out_root / "_sitecustomize"
    _write_sitecustomize(site_dir)

    runs: List[RunSpec] = [
        RunSpec(name="baseline_fp32", amp_dtype=None, tf32=False, flash_sdp=False),
        RunSpec(name="amp_bf16", amp_dtype="bfloat16", tf32=False, flash_sdp=False),
        RunSpec(name="amp_bf16_tf32_flashsdp", amp_dtype="bfloat16", tf32=True, flash_sdp=True),
        RunSpec(name="amp_fp16_tf32_flashsdp", amp_dtype="float16", tf32=True, flash_sdp=True),
    ]

    results: List[RunResult] = []
    results_jsonl = out_root / "results.jsonl"
    results_csv = out_root / "results.csv"

    for spec in runs:
        run_dir = out_root / spec.name
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "lerobot-eval",
            f"--output_dir={str(run_dir)}",
            f"--policy.path={args.policy_path}",
            "--env.type=libero",
            f"--env.task={args.tasks}",
            f"--env.control_mode={args.control_mode}",
            f"--env.episode_length={args.episode_length}",
            f"--eval.batch_size={args.batch_size}",
            f"--eval.n_episodes={args.n_episodes}",
            f"--seed={args.seed}",
            f"--env.max_parallel_tasks={args.max_parallel_tasks}",
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(site_dir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["LEROBOT_OUTPUT_DIR"] = str(run_dir)
        env["LEROBOT_TF32"] = "1" if spec.tf32 else "0"
        env["LEROBOT_FLASH_SDP"] = "1" if spec.flash_sdp else "0"
        env["LEROBOT_MEM_EFF_SDP"] = "1" if spec.mem_efficient_sdp else "0"
        env["LEROBOT_MATH_SDP"] = "1" if spec.math_sdp else "0"
        env["LEROBOT_AMP_DTYPE"] = spec.amp_dtype or ""

        print("\n==============================")
        print(f"RUN: {spec.name}")
        print("CMD:", " ".join(cmd))
        print(
            "ENV: LEROBOT_TF32=%s LEROBOT_FLASH_SDP=%s LEROBOT_AMP_DTYPE=%s"
            % (env["LEROBOT_TF32"], env["LEROBOT_FLASH_SDP"], env["LEROBOT_AMP_DTYPE"] or "none")
        )
        print("==============================")

        rc, out, err, wall = _run_cmd(cmd, env=env)

        (run_dir / "stdout.log").write_text(out, encoding="utf-8")
        (run_dir / "stderr.log").write_text(err, encoding="utf-8")

        ok = (rc == 0)
        success_rate, parse_notes = _parse_success_rate(run_dir) if ok else (None, ["Run failed, skipping success parse."])
        for n in parse_notes[:3]:
            print(f"[parse] {n}")

        eps_per_s = (args.n_episodes / wall) if wall > 0 else None
        steps_total = args.n_episodes * args.episode_length
        steps_per_s = (steps_total / wall) if wall > 0 else None

        # Parse policy latency from the injected timing wrapper
        lat_mean, lat_p50, lat_p90 = _parse_policy_latency(run_dir) if ok else (None, None, None)

        res = RunResult(
            name=spec.name,
            ok=ok,
            returncode=rc,
            wall_s=wall,
            episodes_per_s=eps_per_s,
            approx_steps_per_s=steps_per_s,
            success_rate=success_rate,
            policy_latency_mean_ms=lat_mean,
            policy_latency_p50_ms=lat_p50,
            policy_latency_p90_ms=lat_p90,
            details_path=str(run_dir),
            stdout_tail=_tail(out),
            stderr_tail=_tail(err),
            spec=asdict(spec),
        )
        results.append(res)

        with results_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(res), ensure_ascii=False) + "\n")

    with results_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "ok",
                "returncode",
                "wall_s",
                "episodes_per_s",
                "approx_steps_per_s",
                "success_rate",
                "policy_latency_mean_ms",
                "policy_latency_p50_ms",
                "policy_latency_p90_ms",
                "details_path",
            ],
        )
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "name": r.name,
                    "ok": r.ok,
                    "returncode": r.returncode,
                    "wall_s": f"{r.wall_s:.3f}",
                    "episodes_per_s": "" if r.episodes_per_s is None else f"{r.episodes_per_s:.6f}",
                    "approx_steps_per_s": "" if r.approx_steps_per_s is None else f"{r.approx_steps_per_s:.3f}",
                    "success_rate": "" if r.success_rate is None else f"{r.success_rate:.4f}",
                    "policy_latency_mean_ms": "" if r.policy_latency_mean_ms is None else f"{r.policy_latency_mean_ms:.3f}",
                    "policy_latency_p50_ms": "" if r.policy_latency_p50_ms is None else f"{r.policy_latency_p50_ms:.3f}",
                    "policy_latency_p90_ms": "" if r.policy_latency_p90_ms is None else f"{r.policy_latency_p90_ms:.3f}",
                    "details_path": r.details_path,
                }
            )

    print("\n\n=== SUMMARY ===")
    baseline = next((r for r in results if r.name == "baseline_fp32" and r.ok), None)
    for r in results:
        # Wall time speedup
        wall_speedup = None
        if baseline and baseline.wall_s > 0 and r.wall_s > 0 and r.ok:
            wall_speedup = baseline.wall_s / r.wall_s

        # Policy latency speedup (more accurate for model performance)
        lat_speedup = None
        if (baseline and baseline.policy_latency_mean_ms and baseline.policy_latency_mean_ms > 0
                and r.policy_latency_mean_ms and r.policy_latency_mean_ms > 0 and r.ok):
            lat_speedup = baseline.policy_latency_mean_ms / r.policy_latency_mean_ms

        succ = "" if r.success_rate is None else f"{r.success_rate:.3f}"
        wall_spd = "" if wall_speedup is None else f"{wall_speedup:.2f}x"
        lat_spd = "" if lat_speedup is None else f"{lat_speedup:.2f}x"
        lat_ms = "" if r.policy_latency_mean_ms is None else f"{r.policy_latency_mean_ms:.1f}"
        print(
            f"{r.name:>25} | ok={str(r.ok):5} | wall={r.wall_s:7.2f}s"
            f" | lat_ms={lat_ms:>7}"
            f" | succ={succ:>6} | wall_spd={wall_spd:>6} | lat_spd={lat_spd:>6}"
        )

    print(f"\nWrote: {results_jsonl}")
    print(f"Wrote: {results_csv}")
    return 0 if all(r.ok for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
