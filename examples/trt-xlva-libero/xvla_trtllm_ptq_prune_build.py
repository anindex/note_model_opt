"""
xvla_trtllm_ptq_prune_build.py

Build script for X-VLA (2toINF/X-VLA-Libero) that supports:

1) Optional 2:4 structured pruning (scope: transformer or all)
2) ModelOpt PTQ (fp8 or int8) driven by offline calib_*.npz files
3) Two export modes:

   A) export_mode=transformer_state (recommended for transformer-only PTQ)
      - Writes a normal HF checkpoint (unquantized weights) to:
          out_dir/quant_<quant>_hf/
      - Writes ModelOpt state for ONLY the transformer to:
          out_dir/quant_<quant>_hf/modelopt_state_transformer.pth
      - No reliance on ModelOpt save_pretrained hooks, avoids:
          AssertionError("Model has modelopt state but not the root!")

      Loading then is a two-step:
        - AutoModel.from_pretrained(out_dir/quant_fp8_hf, trust_remote_code=True)
        - restore ModelOpt state into model.transformer

   B) export_mode=modelopt_plugin (only if ptq_scope=full)
      - Uses ModelOpt HF save/restore plugin to write modelopt_state.pth
      - Requires quantizing the root model, not just a submodule

Calib .npz is expected to contain:
  input_ids   : [B, L] int32/int64
  image_input : [B, 2, 3, H, W] float16/float32
  image_mask  : [B, 2] bool/int
  domain_id   : [B] int32/int64
  proprio     : [B, D] float16/float32

Example:
  python xvla_trtllm_ptq_prune_build.py \
    --model_id 2toINF/X-VLA-Libero \
    --calib_dir ./xvla_calib_libero_hf \
    --out_dir ./xvla_opt_out \
    --dtype bf16 \
    --do_prune --prune_scope transformer \
    --do_quant --quant fp8 --ptq_scope transformer \
    --export_mode transformer_state \
    --calib_max_files 16 --denoise_steps 1
"""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel


# -----------------------------
# Warnings and safe patches
# -----------------------------

def quiet_warnings() -> None:
    warnings.filterwarnings("ignore", message="UnsupportedFieldAttributeWarning")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*not tested with nvidia-modelopt.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*_load_state_dict_into_zero3_model.*")


def patch_transformers_sdpa_defaults() -> None:
    """
    Some remote-code models (Florence2 inside X-VLA) may not define _supports_sdpa,
    but newer transformers may try to read it during __init__.
    Make it safe by providing a default on the base class if missing.
    """
    if not hasattr(PreTrainedModel, "_supports_sdpa"):
        PreTrainedModel._supports_sdpa = False
    if not hasattr(PreTrainedModel, "_supports_flash_attn_2"):
        PreTrainedModel._supports_flash_attn_2 = False
    if not hasattr(PreTrainedModel, "_supports_flex_attn"):
        PreTrainedModel._supports_flex_attn = False


# -----------------------------
# Tokenizer loading (offline-friendly)
# -----------------------------

def load_tokenizer(tokenizer_repo: str, local_files_only: bool) -> Optional[object]:
    """
    X-VLA uses a custom config, AutoTokenizer may fail.
    We try AutoTokenizer first, else fall back to downloading tokenizer.json
    and building a fast tokenizer from it.
    """
    try:
        return AutoTokenizer.from_pretrained(
            tokenizer_repo,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
    except Exception:
        pass

    try:
        tok_json = hf_hub_download(
            repo_id=tokenizer_repo,
            filename="tokenizer.json",
            local_files_only=local_files_only,
        )
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file=tok_json)
        if tok.pad_token is None:
            if tok.eos_token is not None:
                tok.pad_token = tok.eos_token
            else:
                tok.add_special_tokens({"pad_token": "<|pad|>"})
        return tok
    except Exception:
        return None


# -----------------------------
# Calibration files
# -----------------------------

def list_calib_files(calib_dir: Path, max_files: int) -> List[Path]:
    files = sorted(calib_dir.glob("calib_*.npz"))
    if not files:
        raise FileNotFoundError(f"No calib_*.npz found in {calib_dir}")
    return files[:max_files] if max_files > 0 else files


def load_npz_batch(p: Path, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    arr = np.load(p)

    def get(name: str) -> np.ndarray:
        if name not in arr:
            raise KeyError(f"{p} missing key: {name}")
        return arr[name]

    input_ids = torch.from_numpy(get("input_ids")).to(device=device, dtype=torch.long)

    image_input = torch.from_numpy(get("image_input")).to(device=device)
    # Keep image in fp32 for Florence2 stability unless you explicitly know it supports bf16 end-to-end
    image_input = image_input.to(dtype=torch.float32)

    image_mask_np = get("image_mask")
    image_mask = torch.from_numpy(image_mask_np).to(device=device)
    if image_mask.dtype != torch.bool:
        image_mask = image_mask > 0

    domain_id = torch.from_numpy(get("domain_id")).to(device=device, dtype=torch.long)

    proprio = torch.from_numpy(get("proprio")).to(device=device, dtype=dtype)
    if proprio.ndim > 2:
        proprio = proprio.reshape(proprio.shape[0], -1)

    return {
        "input_ids": input_ids,
        "image_input": image_input,
        "image_mask": image_mask,
        "domain_id": domain_id,
        "proprio": proprio,
    }


def sanitize_xvla_batch(model: nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    X-VLA action_space.preprocess may assume proprio has action_dim and gripper_idx is valid.
    Calib proprio can be smaller. We pad/truncate to action_dim and clamp input_ids to vocab.
    """
    # input_ids clamp
    vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)
    if vocab_size > 0:
        batch["input_ids"] = torch.clamp(batch["input_ids"], 0, vocab_size - 1)

    if not hasattr(model, "action_space"):
        return batch

    action_dim = int(getattr(model.action_space, "dim_action", 0) or 0)
    if action_dim <= 0:
        return batch

    gi = getattr(model.action_space, "gripper_idx", ())
    if isinstance(gi, int):
        gi = (gi,)
    gi = tuple(int(i) for i in gi)
    safe = tuple(i for i in gi if 0 <= i < action_dim)
    if not safe:
        safe = (action_dim - 1,)
    model.action_space.gripper_idx = safe

    proprio = batch["proprio"]
    B = proprio.shape[0]
    if proprio.shape[1] != action_dim:
        fixed = torch.zeros((B, action_dim), device=proprio.device, dtype=proprio.dtype)
        n = min(action_dim, proprio.shape[1])
        fixed[:, :n] = proprio[:, :n]
        batch["proprio"] = fixed

    return batch


# -----------------------------
# Pruning: 2:4 structured on last dim
# -----------------------------

def prune_2to4_lastdim(w: torch.Tensor) -> torch.Tensor:
    """Apply 2:4 structured sparsity along the output (first) dimension.
    
    For nn.Linear.weight with shape [out_features, in_features], TensorRT's
    sparse tensor cores expect 2:4 sparsity along the M (output) dimension.
    We group every 4 output channels and zero the 2 smallest magnitudes.
    """
    if w.numel() == 0:
        return w
    if w.ndim != 2 or (w.shape[0] % 4) != 0:
        raise ValueError("weight must be 2D with first dim divisible by 4")

    out_dim, in_dim = w.shape
    w4 = w.reshape(out_dim // 4, 4, in_dim)  # [out//4, 4, in]

    # zero 2 smallest magnitude in each group of 4 along dim=1
    idx = torch.argsort(w4.abs(), dim=1)[..., :2, :]  # [out//4, 2, in]
    mask = torch.ones_like(w4, dtype=torch.bool)
    mask.scatter_(dim=1, index=idx, value=False)
    pruned = torch.where(mask, w4, torch.zeros_like(w4))
    return pruned.reshape_as(w)


@torch.inference_mode()
def apply_2to4_pruning(model: nn.Module, scope: str) -> Dict[str, float]:
    if scope not in ("transformer", "all"):
        raise ValueError("scope must be transformer or all")

    roots: List[nn.Module]
    if scope == "transformer":
        if not hasattr(model, "transformer"):
            raise AttributeError("Model has no .transformer attribute")
        roots = [model.transformer]
    else:
        roots = [model]

    pruned_layers = 0
    skipped_layers = 0

    for root in roots:
        for _, m in root.named_modules():
            if isinstance(m, nn.Linear) and isinstance(m.weight, torch.Tensor):
                try:
                    m.weight.data = prune_2to4_lastdim(m.weight.data)
                    pruned_layers += 1
                except ValueError:
                    skipped_layers += 1

    total_elems = 0
    total_zeros = 0
    for p in model.parameters():
        if not p.is_floating_point():
            continue
        total_elems += p.numel()
        total_zeros += int((p == 0).sum().item())

    return {
        "sparsity": float(total_zeros) / float(max(1, total_elems)),
        "total_elems": float(total_elems),
        "total_zeros": float(total_zeros),
        "pruned_layers": float(pruned_layers),
        "skipped_layers": float(skipped_layers),
    }


# -----------------------------
# Critical patch: cast transformer side inputs to match proj layer dtype
# -----------------------------

def patch_transformer_input_cast(xvla_model: nn.Module) -> None:
    """
    Fixes:
      RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
    by casting vlm_features and aux_visual_inputs to match the proj layers.
    """
    if not hasattr(xvla_model, "transformer"):
        return

    tr = xvla_model.transformer
    if not hasattr(tr, "forward"):
        return

    orig_forward = tr.forward

    def _dtype_of(mod: nn.Module, fallback: torch.dtype) -> torch.dtype:
        for p in mod.parameters(recurse=False):
            return p.dtype
        return fallback

    def wrapped_forward(*args, **kwargs):
        # Try kwargs first (most stable)
        if "vlm_features" in kwargs and torch.is_tensor(kwargs["vlm_features"]) and hasattr(tr, "vlm_proj"):
            tgt = _dtype_of(tr.vlm_proj, kwargs["vlm_features"].dtype)
            kwargs["vlm_features"] = kwargs["vlm_features"].to(dtype=tgt)

        if "aux_visual_inputs" in kwargs and torch.is_tensor(kwargs["aux_visual_inputs"]) and hasattr(tr, "aux_visual_proj"):
            tgt = _dtype_of(tr.aux_visual_proj, kwargs["aux_visual_inputs"].dtype)
            kwargs["aux_visual_inputs"] = kwargs["aux_visual_inputs"].to(dtype=tgt)

        return orig_forward(*args, **kwargs)

    tr.forward = wrapped_forward


# -----------------------------
# Model loading and saving
# -----------------------------

def load_xvla_model(model_id: str, torch_dtype: torch.dtype, local_files_only: bool) -> nn.Module:
    # Prefer dtype= (newer transformers), fallback to torch_dtype=
    try:
        return AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=torch_dtype,
            local_files_only=local_files_only,
        )
    except TypeError:
        return AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )


def hf_save_checkpoint(model: nn.Module, tokenizer: Optional[object], export_dir: Path) -> None:
    """
    Save a plain HF checkpoint (no ModelOpt plugin required).
    This is called BEFORE we import modelopt.torch.opt, to avoid any global monkey patches.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(export_dir), safe_serialization=True)
    if tokenizer is not None:
        try:
            tokenizer.save_pretrained(str(export_dir))
        except Exception:
            pass


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


# -----------------------------
# ModelOpt PTQ
# -----------------------------

def get_modelopt_cfg(quant: str):
    import modelopt.torch.quantization as mtq
    q = quant.lower()
    if q == "fp8":
        return mtq.FP8_DEFAULT_CFG
    if q == "int8":
        return mtq.INT8_DEFAULT_CFG
    raise ValueError("quant must be fp8 or int8")


@torch.inference_mode()
def modelopt_ptq_transformer_only(
    model: nn.Module,
    calib_files: List[Path],
    quant: str,
    denoise_steps: int,
    use_autocast: bool,
) -> object:
    """
    Quantize ONLY model.transformer. Returns the transformer-only ModelOpt state object.
    We run calibration by exercising model.generate_actions while stubbing forward_vlm.
    """
    import modelopt.torch.quantization as mtq

    if not hasattr(model, "transformer"):
        raise AttributeError("Model has no .transformer attribute")

    cfg = get_modelopt_cfg(quant)
    device = next(model.parameters()).device
    tr_dtype = next(model.transformer.parameters()).dtype

    # Make sure transformer will not choke on fp32 vlm features
    patch_transformer_input_cast(model)

    def forward_loop(_ignored_module: nn.Module):
        for p in calib_files:
            batch = load_npz_batch(p, device=device, dtype=tr_dtype)
            batch = sanitize_xvla_batch(model, batch)

            # Compute VLM features once in fp32
            enc = model.forward_vlm(batch["input_ids"], batch["image_input"], batch["image_mask"])
            if torch.is_tensor(enc):
                enc = enc.detach()

            # Stub forward_vlm so generate_actions does not re-run Florence2 during calibration
            orig_forward_vlm = model.forward_vlm
            model.forward_vlm = lambda *args, **kwargs: enc
            try:
                if use_autocast and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=tr_dtype):
                        _ = model.generate_actions(**batch, steps=denoise_steps)
                else:
                    _ = model.generate_actions(**batch, steps=denoise_steps)
            finally:
                model.forward_vlm = orig_forward_vlm

            torch.cuda.synchronize()

    # Quantize the transformer module in place
    _ = mtq.quantize(model.transformer, cfg, forward_loop=forward_loop)

    # Export ModelOpt state for transformer only
    import modelopt.torch.opt as mto
    return mto.modelopt_state(model.transformer)


@torch.inference_mode()
def modelopt_ptq_full_model(
    model: nn.Module,
    calib_files: List[Path],
    quant: str,
    denoise_steps: int,
    use_autocast: bool,
) -> object:
    """
    Quantize the full model. This is only needed if you want to use ModelOpt HF plugin export.
    """
    import modelopt.torch.quantization as mtq

    cfg = get_modelopt_cfg(quant)
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    patch_transformer_input_cast(model)

    def forward_loop(_ignored_module: nn.Module):
        for p in calib_files:
            batch = load_npz_batch(p, device=device, dtype=model_dtype)
            batch = sanitize_xvla_batch(model, batch)

            if use_autocast and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=model_dtype):
                    _ = model.generate_actions(**batch, steps=denoise_steps)
            else:
                _ = model.generate_actions(**batch, steps=denoise_steps)

            torch.cuda.synchronize()

    _ = mtq.quantize(model, cfg, forward_loop=forward_loop)

    import modelopt.torch.opt as mto
    return mto.modelopt_state(model)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_id", type=str, default="2toINF/X-VLA-Libero")
    ap.add_argument("--tokenizer_repo", type=str, default="", help="Defaults to model_id")

    ap.add_argument("--calib_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])

    ap.add_argument("--local_files_only", action="store_true")

    ap.add_argument("--do_prune", action="store_true")
    ap.add_argument("--prune_scope", type=str, default="transformer", choices=["transformer", "all"])

    ap.add_argument("--do_quant", action="store_true")
    ap.add_argument("--quant", type=str, default="fp8", choices=["fp8", "int8"])
    ap.add_argument("--ptq_scope", type=str, default="transformer", choices=["transformer", "full"])
    ap.add_argument(
        "--export_mode",
        type=str,
        default="transformer_state",
        choices=["transformer_state", "modelopt_plugin"],
        help="transformer_state is recommended for transformer-only PTQ",
    )

    ap.add_argument("--calib_max_files", type=int, default=16)
    ap.add_argument("--denoise_steps", type=int, default=1)
    ap.add_argument("--use_autocast", action="store_true", help="Wrap generate_actions in autocast during PTQ")

    args = ap.parse_args()

    quiet_warnings()
    patch_transformers_sdpa_defaults()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    calib_dir = Path(args.calib_dir)

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    tok_repo = args.tokenizer_repo.strip() or args.model_id
    tokenizer = load_tokenizer(tok_repo, local_files_only=args.local_files_only)

    # Load model
    model = load_xvla_model(args.model_id, torch_dtype=torch_dtype, local_files_only=args.local_files_only)
    model.to(device)
    model.eval()

    # Always keep Florence2 side stable
    if hasattr(model, "vlm"):
        try:
            model.vlm.to(torch.float32)
        except Exception:
            pass

    patch_transformer_input_cast(model)

    report: Dict[str, object] = {
        "model_id": args.model_id,
        "tokenizer_repo": tok_repo,
        "device": args.device,
        "dtype": args.dtype,
        "local_files_only": bool(args.local_files_only),
        "calib_dir": str(calib_dir),
        "calib_max_files": int(args.calib_max_files),
        "denoise_steps": int(args.denoise_steps),
        "did_prune": False,
        "prune_scope": None,
        "prune_report": None,
        "did_quant": False,
        "quant": None,
        "ptq_scope": None,
        "export_mode": args.export_mode,
        "calib_files_used": [],
        "outputs": {},
    }

    # Optional pruning
    if args.do_prune:
        pr = apply_2to4_pruning(model, scope=args.prune_scope)
        report["did_prune"] = True
        report["prune_scope"] = args.prune_scope
        report["prune_report"] = pr

        pruned_dir = out_dir / "pruned_hf"
        hf_save_checkpoint(model, tokenizer, pruned_dir)
        report["outputs"]["pruned_hf"] = str(pruned_dir)

    # Optional PTQ
    if args.do_quant:
        calib_files = list_calib_files(calib_dir, max_files=args.calib_max_files)
        report["calib_files_used"] = [p.name for p in calib_files]

        q_dir = out_dir / f"quant_{args.quant}_hf"

        # Export a plain HF checkpoint first (no ModelOpt state yet)
        # This avoids the "modelopt state but not the root" assertion when quantizing submodules.
        hf_save_checkpoint(model, tokenizer, q_dir)

        # Run PTQ and extract ModelOpt state
        if args.ptq_scope == "transformer":
            report["ptq_scope"] = "transformer"
            state_obj = modelopt_ptq_transformer_only(
                model=model,
                calib_files=calib_files,
                quant=args.quant,
                denoise_steps=args.denoise_steps,
                use_autocast=bool(args.use_autocast),
            )

            # Save transformer-only state
            import torch as _torch
            _torch.save(state_obj, q_dir / "modelopt_state_transformer.pth")

            (q_dir / "MODELOPT_SCOPE.txt").write_text("transformer_only\n")

        else:
            report["ptq_scope"] = "full"
            state_obj = modelopt_ptq_full_model(
                model=model,
                calib_files=calib_files,
                quant=args.quant,
                denoise_steps=args.denoise_steps,
                use_autocast=bool(args.use_autocast),
            )

            if args.export_mode == "modelopt_plugin":
                # Use ModelOpt HF plugin export for full-model conversion
                from modelopt.torch.opt.plugins.huggingface import enable_huggingface_checkpointing
                enable_huggingface_checkpointing()
                model.save_pretrained(str(q_dir), safe_serialization=True)
                if tokenizer is not None:
                    try:
                        tokenizer.save_pretrained(str(q_dir))
                    except Exception:
                        pass
            else:
                import torch as _torch
                _torch.save(state_obj, q_dir / "modelopt_state_full.pth")
                (q_dir / "MODELOPT_SCOPE.txt").write_text("full_model_state_only\n")

        report["did_quant"] = True
        report["quant"] = args.quant
        report["outputs"][f"quant_{args.quant}_hf"] = str(q_dir)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
