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

# Check for PyTorch semi-structured sparsity support (PyTorch 2.1+)
_HAS_SEMI_STRUCTURED = False
try:
    from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
    # Ensure the backend is available (requires SM80+ GPU like A100, H100, RTX 30xx+)
    _HAS_SEMI_STRUCTURED = True
except ImportError:
    pass


def prune_2to4_lastdim(w: torch.Tensor) -> torch.Tensor:
    """Apply 2:4 structured sparsity along the input (last/K) dimension.
    
    For nn.Linear.weight with shape [out_features, in_features], TensorRT's
    sparse tensor cores expect 2:4 sparsity along the K (in_features) dimension.
    PyTorch's semi-structured sparsity also requires this pattern.
    
    We group every 4 input features and zero the 2 smallest magnitudes.
    """
    if w.numel() == 0:
        return w
    if w.ndim != 2 or (w.shape[1] % 4) != 0:
        raise ValueError("weight must be 2D with in_features (last dim) divisible by 4")

    out_dim, in_dim = w.shape
    # Reshape to [out, in//4, 4] to group along input dimension
    w4 = w.reshape(out_dim, in_dim // 4, 4)  # [out, in//4, 4]

    # Zero 2 smallest magnitude in each group of 4 along dim=2 (the groups of 4)
    idx = torch.argsort(w4.abs(), dim=2)[..., :2]  # [out, in//4, 2]
    mask = torch.ones_like(w4, dtype=torch.bool)
    mask.scatter_(dim=2, index=idx, value=False)
    pruned = torch.where(mask, w4, torch.zeros_like(w4))
    return pruned.reshape_as(w)


def convert_to_semi_structured(weight: torch.Tensor) -> torch.Tensor:
    """Convert a 2:4 pruned dense weight to PyTorch's sparse semi-structured format.
    
    This uses compressed storage (50% memory reduction) and enables efficient
    sparse GEMM kernels on Ampere+ GPUs (SM80+).
    
    Requirements:
    - PyTorch 2.1+
    - CUDA GPU with SM80+ (A100, H100, RTX 30xx, RTX 40xx, RTX 50xx)
    - Weight shape must be compatible (both dims divisible by 16 for fp16/bf16)
    """
    if not _HAS_SEMI_STRUCTURED:
        raise RuntimeError(
            "PyTorch semi-structured sparsity not available. "
            "Requires PyTorch 2.1+ with CUDA support."
        )
    return to_sparse_semi_structured(weight)


@torch.inference_mode()
def apply_2to4_pruning(
    model: nn.Module,
    scope: str,
    use_semi_structured: bool = False,
    semi_structured_dtype: Optional[torch.dtype] = None,
) -> Dict[str, float]:
    """Apply 2:4 structured pruning to model weights.
    
    Args:
        model: The model to prune
        scope: "transformer" or "all"
        use_semi_structured: If True, convert pruned weights to PyTorch's
            sparse semi-structured format for memory savings and faster inference.
            Requires PyTorch 2.1+ and Ampere+ GPU.
        semi_structured_dtype: Target dtype for semi-structured conversion.
            Must be float16 or bfloat16 (required by cuSPARSELt).
            If None, will use the model's dtype or fall back to bfloat16.
    
    Returns:
        Dict with pruning statistics
    """
    if scope not in ("transformer", "all"):
        raise ValueError("scope must be transformer or all")

    if use_semi_structured and not _HAS_SEMI_STRUCTURED:
        raise RuntimeError(
            "Semi-structured sparsity requested but not available. "
            "Requires PyTorch 2.1+ with CUDA support."
        )

    # Semi-structured sparsity only supports fp16/bf16
    SUPPORTED_SEMI_DTYPES = (torch.float16, torch.bfloat16)

    roots: List[nn.Module]
    if scope == "transformer":
        if not hasattr(model, "transformer"):
            raise AttributeError("Model has no .transformer attribute")
        roots = [model.transformer]
    else:
        roots = [model]

    pruned_layers = 0
    skipped_layers = 0
    semi_structured_layers = 0
    semi_structured_failures = 0
    dtype_converted_layers = 0

    for root in roots:
        for name, m in root.named_modules():
            if isinstance(m, nn.Linear) and isinstance(m.weight, torch.Tensor):
                # Check if in_features (weight.shape[1]) is divisible by 4
                if m.weight.ndim != 2 or (m.weight.shape[1] % 4) != 0:
                    skipped_layers += 1
                    continue
                try:
                    # First apply 2:4 pruning pattern along in_features dimension
                    pruned_weight = prune_2to4_lastdim(m.weight.data)
                    
                    if use_semi_structured:
                        # Try to convert to semi-structured sparse format
                        # This requires specific shape constraints AND fp16/bf16 dtype
                        try:
                            # Semi-structured requires contiguous memory and fp16/bf16
                            pruned_weight = pruned_weight.contiguous()
                            
                            # Convert dtype if needed (cuSPARSELt only supports fp16/bf16)
                            if pruned_weight.dtype not in SUPPORTED_SEMI_DTYPES:
                                target_dtype = semi_structured_dtype
                                if target_dtype is None:
                                    # Default to bf16 as it has better numerical range
                                    target_dtype = torch.bfloat16
                                if target_dtype not in SUPPORTED_SEMI_DTYPES:
                                    target_dtype = torch.bfloat16
                                pruned_weight = pruned_weight.to(dtype=target_dtype)
                                dtype_converted_layers += 1
                            
                            sparse_weight = convert_to_semi_structured(pruned_weight)
                            m.weight = nn.Parameter(sparse_weight, requires_grad=False)
                            semi_structured_layers += 1
                        except Exception as e:
                            # Fallback to dense storage if conversion fails
                            # (e.g., incompatible shapes, GPU not supported, etc.)
                            m.weight.data = pruned_weight
                            semi_structured_failures += 1
                            warnings.warn(
                                f"Semi-structured conversion failed for {name}: {e}. "
                                "Using dense storage."
                            )
                    else:
                        m.weight.data = pruned_weight
                    
                    pruned_layers += 1
                except ValueError:
                    skipped_layers += 1

    total_elems = 0
    total_zeros = 0
    for p in model.parameters():
        if not p.is_floating_point():
            continue
        # Handle semi-structured tensors which may not support == 0 directly
        try:
            if hasattr(p, 'to_dense'):
                dense_p = p.to_dense()
                total_elems += dense_p.numel()
                total_zeros += int((dense_p == 0).sum().item())
            else:
                total_elems += p.numel()
                total_zeros += int((p == 0).sum().item())
        except Exception:
            total_elems += p.numel()

    return {
        "sparsity": float(total_zeros) / float(max(1, total_elems)),
        "total_elems": float(total_elems),
        "total_zeros": float(total_zeros),
        "pruned_layers": float(pruned_layers),
        "skipped_layers": float(skipped_layers),
        "semi_structured_layers": float(semi_structured_layers),
        "semi_structured_failures": float(semi_structured_failures),
        "dtype_converted_layers": float(dtype_converted_layers),
        "use_semi_structured": use_semi_structured,
    }


# -----------------------------
# Critical patches for dtype mismatches
# -----------------------------

def _first_tensor_dtype(mod: nn.Module) -> Optional[torch.dtype]:
    """Get the dtype of the first parameter/buffer in a module."""
    for p in mod.parameters(recurse=False):
        return p.dtype
    for _, b in mod.named_buffers(recurse=False):
        if torch.is_tensor(b):
            return b.dtype
    for p in mod.parameters():
        return p.dtype
    for _, b in mod.named_buffers():
        if torch.is_tensor(b):
            return b.dtype
    return None


def patch_action_encoder_dtype(model: nn.Module) -> None:
    """
    X-VLA action_encoder uses a custom matmul path:
      y = matmul(x, W) + b
    If x and W have different dtypes you get:
      expected scalar type BFloat16 but found Float
    Patch forward to cast x to encoder internal dtype.
    """
    if not (hasattr(model, "transformer") and hasattr(model.transformer, "action_encoder")):
        return

    enc = model.transformer.action_encoder
    
    # Prevent double-patching
    if getattr(enc, "_dtype_patched", False):
        return
    enc._dtype_patched = True
    
    orig_forward = enc.forward

    def wrapped_forward(x: torch.Tensor, *args, **kwargs):
        target = _first_tensor_dtype(enc)
        if target is not None and torch.is_tensor(x) and x.dtype != target:
            x = x.to(dtype=target)
        return orig_forward(x, *args, **kwargs)

    enc.forward = wrapped_forward


def patch_transformer_input_cast(xvla_model: nn.Module, target_dtype: Optional[torch.dtype] = None) -> None:
    """
    Register forward pre-hooks on vlm_proj and aux_visual_proj to cast inputs to the target dtype.
    This handles the case where VLM features are computed in fp32 but transformer weights are in fp16/bf16.
    Uses hooks instead of monkey-patching to be robust to module replacement by ModelOpt.
    """
    if not hasattr(xvla_model, "transformer"):
        return

    tr = xvla_model.transformer
    
    # Determine target dtype
    if target_dtype is None:
        target_dtype = torch.bfloat16
    
    def make_pre_hook(dtype):
        def pre_hook(module, args):
            # args is a tuple of inputs
            if len(args) > 0:
                inp = args[0]
                if torch.is_tensor(inp) and inp.is_floating_point() and inp.dtype != dtype:
                    return (inp.to(dtype=dtype),) + args[1:]
            return args
        return pre_hook
    
    # Register hooks on specific projection layers
    for proj_name in ("vlm_proj", "aux_visual_proj"):
        if not hasattr(tr, proj_name):
            continue
        
        proj = getattr(tr, proj_name)
        
        # Remove existing hooks if any (to prevent duplication)
        if hasattr(proj, "_dtype_cast_hook_handle"):
            proj._dtype_cast_hook_handle.remove()
        
        # Register new pre-hook
        handle = proj.register_forward_pre_hook(make_pre_hook(target_dtype))
        proj._dtype_cast_hook_handle = handle


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
    
    Note: Semi-structured sparse tensors must be converted to dense before saving,
    as safetensors cannot handle the compressed sparse storage format.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert any semi-structured sparse tensors to dense before saving
    # (safetensors cannot serialize SparseSemiStructuredTensor)
    sparse_converted = []
    for name, param in model.named_parameters():
        if hasattr(param, 'to_dense') and hasattr(param, 'compressed_swizzled_bitmask'):
            # This is a SparseSemiStructuredTensor - convert to dense
            dense_data = param.to_dense()
            # We need to find the module and replace the parameter
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, param_name = parts
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                param_name = name
            setattr(parent, param_name, nn.Parameter(dense_data, requires_grad=False))
            sparse_converted.append(name)
    
    if sparse_converted:
        print(f"[save] Converted {len(sparse_converted)} semi-structured sparse tensors to dense for serialization")
    
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
    target_dtype: torch.dtype = torch.bfloat16,
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
    tr_dtype = target_dtype  # Use explicit dtype instead of detecting

    patch_action_encoder_dtype(model)

    def forward_loop(_ignored_module: nn.Module):
        # Apply input cast patch inside forward_loop
        # This ensures it's applied AFTER mtq.quantize modifies the module
        patch_transformer_input_cast(model, target_dtype=tr_dtype)
        
        for p in calib_files:
            batch = load_npz_batch(p, device=device, dtype=tr_dtype)
            batch = sanitize_xvla_batch(model, batch)

            # Run VLM once to get features, but keep them in the stub
            # We'll use autocast to handle the VLM in fp32 internally
            with torch.no_grad():
                # VLM expects fp32 image_input, compute features
                enc = model.forward_vlm(batch["input_ids"], batch["image_input"], batch["image_mask"])
            
            # Cast VLM output to target dtype and clone to avoid any aliasing issues
            if isinstance(enc, dict):
                enc_casted = {}
                for k, v in enc.items():
                    if torch.is_tensor(v) and v.is_floating_point():
                        enc_casted[k] = v.detach().clone().to(dtype=tr_dtype)
                    else:
                        enc_casted[k] = v
                enc = enc_casted
            elif torch.is_tensor(enc):
                enc = enc.detach().clone().to(dtype=tr_dtype)

            # Stub forward_vlm to bypass VLM and return pre-computed bf16 features
            orig_forward_vlm = model.forward_vlm
            
            # Create stub as a class to avoid closure issues
            class VLMStub:
                def __init__(self, cached_enc):
                    self.cached_enc = cached_enc
                def __call__(self, *args, **kwargs):
                    return self.cached_enc
            
            model.forward_vlm = VLMStub(enc)
            
            try:
                # Always use autocast for the transformer part
                with torch.autocast(device_type="cuda", dtype=tr_dtype, enabled=True):
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
    target_dtype: torch.dtype = torch.bfloat16,
) -> object:
    """
    Quantize the full model. This is only needed if you want to use ModelOpt HF plugin export.
    """
    import modelopt.torch.quantization as mtq

    cfg = get_modelopt_cfg(quant)
    device = next(model.parameters()).device
    model_dtype = target_dtype  # Use explicit dtype

    patch_transformer_input_cast(model, target_dtype=target_dtype)
    patch_action_encoder_dtype(model)

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
    ap.add_argument(
        "--prune_semi_structured",
        action="store_true",
        help="Convert pruned weights to PyTorch semi-structured sparse format. "
             "Reduces memory by 50%% and enables fast sparse GEMM kernels. "
             "Requires PyTorch 2.1+ and Ampere+ GPU (SM80+).",
    )

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

    # Apply dtype patches with explicit target dtype
    patch_transformer_input_cast(model, target_dtype=torch_dtype)
    patch_action_encoder_dtype(model)

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
        pr = apply_2to4_pruning(
            model,
            scope=args.prune_scope,
            use_semi_structured=args.prune_semi_structured,
            semi_structured_dtype=torch_dtype,  # Use the model's target dtype (bf16/fp16)
        )
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
                target_dtype=torch_dtype,
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
                target_dtype=torch_dtype,
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
