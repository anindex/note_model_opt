"""
Benchmark SmolVLA baseline vs Pruna-optimized models.

Measures latency (via select_action), peak GPU memory, and param count.

Examples:
  python bench_smolvla_pruna.py --smashed-dir ./out/torchao_int8wo

  # Multiple variants
  python bench_smolvla_pruna.py --smashed-dir ./out/torchao_int8wo ./out/half_fp16

  # More warmup for stable numbers
  python bench_smolvla_pruna.py --smashed-dir ./out/torchao_int8wo --warmup-iters 50
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class BenchmarkResult:
    """Stores benchmark numbers for one model."""
    name: str
    params: int
    latency_mean_ms: float
    latency_std_ms: float
    latency_min_ms: float
    latency_max_ms: float
    memory_peak_mb: float
    device: str


def pick_device(user_device: str | None) -> str:
    """Pick CUDA > MPS > CPU if user didn't specify."""
    if user_device:
        return user_device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def count_parameters(model: torch.nn.Module) -> int:
    """Count all params (trainable + frozen)."""
    return sum(p.numel() for p in model.parameters())


def create_dummy_observation_from_model(
    model: torch.nn.Module,
    device: str,
    batch_size: int = 1,
    task: str = "pick up the object",
) -> dict[str, torch.Tensor]:
    """
    Build a dummy observation dict by reading the model's config.
    Also tokenizes the task string (SmolVLA needs language tokens).
    """
    obs = {}
    
    # Get input features from model config
    if hasattr(model, 'config') and hasattr(model.config, 'input_features'):
        input_features = model.config.input_features
        for key, feature in input_features.items():
            shape = feature.shape
            obs[key] = torch.randn(
                batch_size, *shape,
                device=device, dtype=torch.float32,
            )
    else:
        # Fallback: SmolVLA default format (camera1, camera2, camera3 + state)
        print("  Warning: Could not read model config, using SmolVLA defaults")
        for cam_idx in [1, 2, 3]:
            key = f"observation.images.camera{cam_idx}"
            obs[key] = torch.randn(
                batch_size, 3, 256, 256,
                device=device, dtype=torch.float32,
            )
        obs["observation.state"] = torch.randn(
            batch_size, 6,
            device=device, dtype=torch.float32,
        )
    
    # Add language tokens for SmolVLA
    # SmolVLA requires tokenized task instruction
    if hasattr(model, 'config') and hasattr(model.config, 'vlm_model_name'):
        try:
            from transformers import AutoTokenizer
            vlm_name = model.config.vlm_model_name
            max_length = getattr(model.config, 'tokenizer_max_length', 48)
            
            tokenizer = AutoTokenizer.from_pretrained(vlm_name)
            tokens = tokenizer(
                task,
                return_tensors='pt',
                padding='max_length',
                max_length=max_length,
                truncation=True,
            )
            # SmolVLA expects observation.language.tokens and attention_mask (bool)
            obs["observation.language.tokens"] = tokens['input_ids'].to(device).expand(batch_size, -1)
            obs["observation.language.attention_mask"] = tokens['attention_mask'].bool().to(device).expand(batch_size, -1)
        except Exception as e:
            print(f"  Warning: Could not tokenize task: {e}")
            # Fallback: dummy tokens
            obs["observation.language.tokens"] = torch.zeros(
                batch_size, 48, dtype=torch.long, device=device
            )
            obs["observation.language.attention_mask"] = torch.ones(
                batch_size, 48, dtype=torch.bool, device=device
            )
    
    return obs


def benchmark_model(
    model: torch.nn.Module,
    observation: dict[str, torch.Tensor],
    name: str,
    device: str,
    warmup_iters: int = 10,
    bench_iters: int = 50,
) -> BenchmarkResult:
    """Run warmup + timed inference, return latency/memory stats."""
    model = model.to(device).eval()
    
    # Reset memory stats
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    # Warmup (critical for torch.compile models)
    print(f"  [{name}] Warming up ({warmup_iters} iters)...", end="", flush=True)
    with torch.no_grad():
        for _ in range(warmup_iters):
            # Use select_action for inference (LeRobot policy interface)
            _ = model.select_action(observation)
    
    if device == "cuda":
        torch.cuda.synchronize()
    print(" done")
    
    # Benchmark
    print(f"  [{name}] Benchmarking ({bench_iters} iters)...", end="", flush=True)
    latencies = []
    
    with torch.no_grad():
        for _ in range(bench_iters):
            if device == "cuda":
                torch.cuda.synchronize()
            
            t0 = time.perf_counter()
            _ = model.select_action(observation)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms
    
    print(" done")
    
    # Memory stats
    if device == "cuda":
        memory_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        memory_peak_mb = 0.0  # Can't easily measure on CPU/MPS
    
    latencies = np.array(latencies)
    
    return BenchmarkResult(
        name=name,
        params=count_parameters(model),
        latency_mean_ms=float(np.mean(latencies)),
        latency_std_ms=float(np.std(latencies)),
        latency_min_ms=float(np.min(latencies)),
        latency_max_ms=float(np.max(latencies)),
        memory_peak_mb=memory_peak_mb,
        device=device,
    )


def load_baseline_model(model_id: str, device: str) -> torch.nn.Module:
    """Load vanilla SmolVLA from HuggingFace."""
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError:
        # Fallback for older lerobot versions
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    
    print(f"[bench] Loading baseline: {model_id}")
    policy = SmolVLAPolicy.from_pretrained(model_id)
    return policy.to(device).eval()


def load_smashed_model(smashed_dir: str | Path, device: str) -> torch.nn.Module:
    """Load a Pruna-optimized model from disk."""
    from pruna import PrunaModel
    
    smashed_dir = Path(smashed_dir)
    print(f"[bench] Loading smashed model: {smashed_dir}")
    
    # Try PrunaModel.from_pretrained first
    try:
        model = PrunaModel.from_pretrained(str(smashed_dir))
        return model.to(device).eval()
    except Exception as e:
        print(f"  Warning: PrunaModel.from_pretrained failed: {e}")
    
    # Fallback: try loading as regular PyTorch model
    try:
        try:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        except ImportError:
            from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        model = SmolVLAPolicy.from_pretrained(str(smashed_dir))
        return model.to(device).eval()
    except Exception as e:
        print(f"  Warning: SmolVLAPolicy.from_pretrained failed: {e}")
    
    raise RuntimeError(f"Could not load model from {smashed_dir}")


def print_results_table(results: list[BenchmarkResult], baseline_result: BenchmarkResult) -> None:
    """Print a nice comparison table."""
    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS")
    print("=" * 90)
    
    # Header
    print(f"{'Model':<30} {'Params':>12} {'Latency (ms)':>16} {'Memory (MB)':>12} {'Speedup':>10}")
    print(f"{'':30} {'':>12} {'mean ± std':>16} {'peak':>12} {'vs base':>10}")
    print("-" * 90)
    
    # Baseline
    print(
        f"{baseline_result.name:<30} "
        f"{baseline_result.params:>12,} "
        f"{baseline_result.latency_mean_ms:>7.2f} ± {baseline_result.latency_std_ms:<5.2f} "
        f"{baseline_result.memory_peak_mb:>12.1f} "
        f"{'1.00x':>10}"
    )
    
    # Smashed variants
    for r in results:
        speedup = baseline_result.latency_mean_ms / r.latency_mean_ms if r.latency_mean_ms > 0 else 0
        param_reduction = (1 - r.params / baseline_result.params) * 100 if baseline_result.params > 0 else 0
        memory_reduction = (1 - r.memory_peak_mb / baseline_result.memory_peak_mb) * 100 if baseline_result.memory_peak_mb > 0 else 0
        
        print(
            f"{r.name:<30} "
            f"{r.params:>12,} "
            f"{r.latency_mean_ms:>7.2f} ± {r.latency_std_ms:<5.2f} "
            f"{r.memory_peak_mb:>12.1f} "
            f"{speedup:>9.2f}x"
        )
    
    print("=" * 90)
    
    # Summary
    if results:
        best_speedup = max(baseline_result.latency_mean_ms / r.latency_mean_ms for r in results)
        best_memory = min(r.memory_peak_mb for r in results)
        print(f"\nBest speedup: {best_speedup:.2f}x")
        if baseline_result.memory_peak_mb > 0:
            memory_savings = (1 - best_memory / baseline_result.memory_peak_mb) * 100
            print(f"Best memory savings: {memory_savings:.1f}%")


def export_csv(
    results: list[BenchmarkResult],
    baseline_result: BenchmarkResult,
    csv_path: str | Path,
) -> None:
    """Dump results to CSV for further analysis."""
    import csv
    
    csv_path = Path(csv_path)
    
    all_results = [baseline_result] + results
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "params", "latency_mean_ms", "latency_std_ms",
            "latency_min_ms", "latency_max_ms", "memory_peak_mb", "device",
        ])
        for r in all_results:
            writer.writerow([
                r.name, r.params, r.latency_mean_ms, r.latency_std_ms,
                r.latency_min_ms, r.latency_max_ms, r.memory_peak_mb, r.device,
            ])
    
    print(f"\n[bench] Results exported to: {csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Benchmark SmolVLA: baseline vs Pruna-optimized variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    ap.add_argument(
        "--baseline-id",
        default="lerobot/smolvla_base",
        help="HuggingFace model ID for baseline (default: lerobot/smolvla_base)",
    )
    ap.add_argument(
        "--smashed-dir",
        nargs="+",
        required=True,
        help="Path(s) to Pruna-smashed model directory(ies)",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="Device: cpu|cuda|mps (default: auto-detect)",
    )
    ap.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    ap.add_argument(
        "--bench-iters",
        type=int,
        default=50,
        help="Number of benchmark iterations (default: 50)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )
    ap.add_argument(
        "--task",
        default="pick up the object",
        help="Task instruction for inference (default: 'pick up the object')",
    )
    ap.add_argument(
        "--export-csv",
        type=str,
        default=None,
        help="Export results to CSV file",
    )
    ap.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline model (use cached results)",
    )
    
    args = ap.parse_args()
    
    device = pick_device(args.device)
    print(f"[bench] Device: {device}")
    print(f"[bench] Warmup iters: {args.warmup_iters}")
    print(f"[bench] Benchmark iters: {args.bench_iters}")
    print(f"[bench] Batch size: {args.batch_size}")
    
    results = []
    baseline_result = None
    observation = None
    
    # Benchmark baseline
    if not args.skip_baseline:
        print("\n" + "-" * 60)
        print("BASELINE MODEL")
        print("-" * 60)
        
        baseline_model = load_baseline_model(args.baseline_id, device)
        
        # Create dummy observation based on model's expected features
        observation = create_dummy_observation_from_model(
            model=baseline_model,
            device=device,
            batch_size=args.batch_size,
            task=args.task,
        )
        print(f"[bench] Observation keys: {list(observation.keys())}")
        
        baseline_result = benchmark_model(
            model=baseline_model,
            observation=observation,
            name="baseline",
            device=device,
            warmup_iters=args.warmup_iters,
            bench_iters=args.bench_iters,
        )
        
        # Free memory
        del baseline_model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
        # Create dummy baseline result for comparison
        baseline_result = BenchmarkResult(
            name="baseline (skipped)",
            params=0,
            latency_mean_ms=1.0,
            latency_std_ms=0.0,
            latency_min_ms=1.0,
            latency_max_ms=1.0,
            memory_peak_mb=0.0,
            device=device,
        )
    
    # Benchmark smashed variants
    for smashed_dir in args.smashed_dir:
        print("\n" + "-" * 60)
        print(f"SMASHED MODEL: {smashed_dir}")
        print("-" * 60)
        
        try:
            smashed_model = load_smashed_model(smashed_dir, device)
            
            # Create observation from smashed model if not already created
            if observation is None:
                observation = create_dummy_observation_from_model(
                    model=smashed_model,
                    device=device,
                    batch_size=args.batch_size,
                    task=args.task,
                )
                print(f"[bench] Observation keys: {list(observation.keys())}")
            
            # Extract name from directory
            name = Path(smashed_dir).name
            
            result = benchmark_model(
                model=smashed_model,
                observation=observation,
                name=name,
                device=device,
                warmup_iters=args.warmup_iters,
                bench_iters=args.bench_iters,
            )
            results.append(result)
            
            # Free memory
            del smashed_model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"  Error benchmarking {smashed_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print results
    print_results_table(results, baseline_result)
    
    # Export if requested
    if args.export_csv:
        export_csv(results, baseline_result, args.export_csv)


if __name__ == "__main__":
    main()