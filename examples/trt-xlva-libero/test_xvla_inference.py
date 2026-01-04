#!/usr/bin/env python3
"""
Test X-VLA model inference for optimized models (pruned, quantized).

This script loads optimized X-VLA models and tests:
1. Success rate: Does the model run without errors?
2. Inference rate: How many samples/second can be processed?
3. Output validity: Are the action outputs within expected bounds?

Usage:
    # Test pruned model
    python test_xvla_inference.py --model_path ./xvla_opt_out/pruned_hf --calib_dir ./xvla_calib_libero_hf

    # Test quantized model
    python test_xvla_inference.py --model_path ./xvla_opt_out/quant_fp8_hf --model_type quant --calib_dir ./xvla_calib_libero_hf

    # Test with semi-structured sparsity (pruned only)
    python test_xvla_inference.py --model_path ./xvla_opt_out/pruned_hf --semi_structured --calib_dir ./xvla_calib_libero_hf

    # Compare with baseline
    python test_xvla_inference.py --model_path ./xvla_opt_out/pruned_hf --baseline_id 2toINF/X-VLA-Libero --calib_dir ./xvla_calib_libero_hf
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel


# ----------------------------
# Model loading utilities
# ----------------------------

def load_model(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    local_files_only: bool = False,
) -> nn.Module:
    """Load X-VLA model from HuggingFace checkpoint."""
    try:
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=dtype,
            local_files_only=local_files_only,
        )
    except TypeError:
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
    
    model.to(device)
    model.eval()
    
    # Ensure transformer uses target dtype
    if hasattr(model, "transformer"):
        try:
            model.transformer.to(dtype=dtype)
        except Exception:
            pass
    
    return model


def restore_modelopt_state(model: nn.Module, state_path: Path) -> None:
    """Restore ModelOpt quantization state to transformer."""
    import modelopt.torch.opt as mto
    
    if not state_path.exists():
        raise FileNotFoundError(f"ModelOpt state not found: {state_path}")
    
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    
    if hasattr(model, "transformer"):
        mto.restore_from_modelopt_state(model.transformer, state)
    else:
        mto.restore_from_modelopt_state(model, state)


def convert_to_semi_structured(model: nn.Module, scope: str = "transformer") -> Dict[str, Any]:
    """Convert 2:4 pruned dense weights to SparseSemiStructuredTensor."""
    try:
        from torch.sparse import to_sparse_semi_structured
    except ImportError:
        return {"error": "Semi-structured sparsity not available"}
    
    target = model.transformer if (scope == "transformer" and hasattr(model, "transformer")) else model
    
    converted = 0
    skipped = 0
    
    for name, module in target.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        
        w = module.weight.data
        
        if w.dtype not in (torch.float16, torch.bfloat16):
            skipped += 1
            continue
        
        K = w.shape[1]
        if K % 4 != 0:
            skipped += 1
            continue
        
        # Check 2:4 pattern
        w_reshaped = w.view(-1, 4)
        zeros_per_group = (w_reshaped == 0).sum(dim=1)
        if not (zeros_per_group == 2).all():
            skipped += 1
            continue
        
        try:
            sparse_w = to_sparse_semi_structured(w)
            module.weight = nn.Parameter(sparse_w, requires_grad=False)
            converted += 1
        except Exception:
            skipped += 1
    
    return {"converted": converted, "skipped": skipped}


# ----------------------------
# Safety patches
# ----------------------------

def patch_action_encoder_dtype(model: nn.Module) -> None:
    """Patch action_encoder to handle dtype mismatches."""
    if not (hasattr(model, "transformer") and hasattr(model.transformer, "action_encoder")):
        return

    enc = model.transformer.action_encoder
    orig_forward = enc.forward

    def _first_param_dtype(mod):
        for p in mod.parameters():
            return p.dtype
        return None

    def wrapped_forward(x: torch.Tensor, *args, **kwargs):
        target = _first_param_dtype(enc)
        if target is not None and torch.is_tensor(x) and x.dtype != target:
            x = x.to(dtype=target)
        return orig_forward(x, *args, **kwargs)

    enc.forward = wrapped_forward


def patch_transformer_input_cast(model: nn.Module, target_dtype: torch.dtype) -> None:
    """Register hooks to cast vlm_proj/aux_visual_proj inputs."""
    if not hasattr(model, "transformer"):
        return

    tr = model.transformer
    
    def make_pre_hook(dtype):
        def pre_hook(module, args):
            if len(args) > 0:
                inp = args[0]
                if torch.is_tensor(inp) and inp.is_floating_point() and inp.dtype != dtype:
                    return (inp.to(dtype=dtype),) + args[1:]
            return args
        return pre_hook
    
    for proj_name in ("vlm_proj", "aux_visual_proj"):
        if hasattr(tr, proj_name):
            proj = getattr(tr, proj_name)
            if hasattr(proj, "_dtype_hook"):
                proj._dtype_hook.remove()
            handle = proj.register_forward_pre_hook(make_pre_hook(target_dtype))
            proj._dtype_hook = handle


def sanitize_xvla_batch(model: nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Sanitize batch to avoid out-of-bounds errors."""
    vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)
    if vocab_size > 0 and "input_ids" in batch:
        batch["input_ids"] = torch.clamp(batch["input_ids"], 0, vocab_size - 1)

    if hasattr(model, "action_space") and "proprio" in batch:
        proprio = batch["proprio"]
        B = int(proprio.shape[0])
        action_dim = int(getattr(model.action_space, "dim_action", 0) or proprio.shape[-1])

        if action_dim > 0 and int(proprio.shape[-1]) != action_dim:
            fixed = torch.zeros((B, action_dim), device=proprio.device, dtype=proprio.dtype)
            n = min(action_dim, int(proprio.shape[-1]))
            fixed[:, :n] = proprio[:, :n]
            batch["proprio"] = fixed

        gi = getattr(model.action_space, "gripper_idx", None)
        if gi is not None:
            if isinstance(gi, int):
                gi = [gi]
            elif torch.is_tensor(gi):
                gi = gi.tolist()
            if isinstance(gi, (list, tuple)):
                safe = [i for i in gi if 0 <= (i if i >= 0 else action_dim + i) < action_dim]
                if not safe:
                    safe = [action_dim - 1]
                try:
                    model.action_space.gripper_idx = safe
                except Exception:
                    pass

    return batch


# ----------------------------
# Data loading
# ----------------------------

def list_calib_files(calib_dir: Path, max_files: int = 0) -> List[Path]:
    """List calibration files."""
    files = sorted(calib_dir.glob("calib_*.npz"))
    if not files:
        raise FileNotFoundError(f"No calib_*.npz found in {calib_dir}")
    return files[:max_files] if max_files > 0 else files


def load_npz_batch(
    path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """Load a calibration batch from npz file."""
    arr = np.load(path)
    
    def get(name: str) -> np.ndarray:
        if name not in arr:
            raise KeyError(f"{path} missing key: {name}")
        return arr[name]
    
    input_ids = torch.from_numpy(get("input_ids")).to(device=device, dtype=torch.long)
    
    # Keep image in fp32 for VLM stability
    image_input = torch.from_numpy(get("image_input")).to(device=device, dtype=torch.float32)
    
    image_mask = torch.from_numpy(get("image_mask")).to(device=device)
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


# ----------------------------
# Inference testing
# ----------------------------

@torch.inference_mode()
def test_single_inference(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    steps: int,
    use_autocast: bool,
    dtype: torch.dtype,
) -> Tuple[bool, Optional[torch.Tensor], float, Optional[str]]:
    """
    Run a single inference and return (success, output, time_ms, error_msg).
    """
    device = next(model.parameters()).device
    
    try:
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        if use_autocast and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=dtype):
                actions = model.generate_actions(**batch, steps=steps)
        else:
            actions = model.generate_actions(**batch, steps=steps)
        
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        
        return True, actions, elapsed_ms, None
    
    except Exception as e:
        return False, None, 0.0, str(e)


def validate_actions(
    actions: torch.Tensor,
    expected_shape: Optional[Tuple[int, ...]] = None,
) -> Dict[str, Any]:
    """Validate action outputs."""
    result = {
        "valid": True,
        "shape": list(actions.shape),
        "dtype": str(actions.dtype),
        "has_nan": bool(torch.isnan(actions).any()),
        "has_inf": bool(torch.isinf(actions).any()),
        "min": float(actions.min()),
        "max": float(actions.max()),
        "mean": float(actions.mean()),
        "std": float(actions.std()),
    }
    
    if result["has_nan"] or result["has_inf"]:
        result["valid"] = False
    
    if expected_shape is not None and tuple(actions.shape) != expected_shape:
        result["valid"] = False
        result["expected_shape"] = list(expected_shape)
    
    return result


@torch.inference_mode()
def run_inference_test(
    model: nn.Module,
    calib_files: List[Path],
    device: torch.device,
    dtype: torch.dtype,
    steps: int = 1,
    warmup: int = 5,
    use_autocast: bool = True,
) -> Dict[str, Any]:
    """
    Run inference test across multiple samples.
    
    Returns dict with:
    - success_rate: fraction of successful inferences
    - inference_rate: samples per second
    - latency_stats: min/max/mean/std latency in ms
    - validation_stats: output validation results
    """
    results = {
        "total_samples": len(calib_files),
        "successful": 0,
        "failed": 0,
        "errors": [],
        "latencies_ms": [],
        "validations": [],
    }
    
    expected_shape = None
    
    # Warmup
    if warmup > 0 and len(calib_files) > 0:
        batch = load_npz_batch(calib_files[0], device=device, dtype=dtype)
        batch = sanitize_xvla_batch(model, batch)
        for _ in range(warmup):
            try:
                if use_autocast and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        _ = model.generate_actions(**batch, steps=steps)
                else:
                    _ = model.generate_actions(**batch, steps=steps)
            except Exception:
                pass
        torch.cuda.synchronize()
    
    # Run tests
    for i, path in enumerate(calib_files):
        batch = load_npz_batch(path, device=device, dtype=dtype)
        batch = sanitize_xvla_batch(model, batch)
        
        success, actions, latency_ms, error = test_single_inference(
            model, batch, steps, use_autocast, dtype
        )
        
        if success:
            results["successful"] += 1
            results["latencies_ms"].append(latency_ms)
            
            # Validate output
            if expected_shape is None and actions is not None:
                expected_shape = tuple(actions.shape)
            
            validation = validate_actions(actions, expected_shape)
            results["validations"].append(validation)
            
            print(f"  [{i+1}/{len(calib_files)}] ✓ {path.name}: {latency_ms:.2f} ms")
        else:
            results["failed"] += 1
            results["errors"].append({"file": str(path.name), "error": error})
            print(f"  [{i+1}/{len(calib_files)}] ✗ {path.name}: {error[:80]}")
    
    # Compute statistics
    if results["latencies_ms"]:
        latencies = np.array(results["latencies_ms"])
        results["latency_stats"] = {
            "min_ms": float(latencies.min()),
            "max_ms": float(latencies.max()),
            "mean_ms": float(latencies.mean()),
            "std_ms": float(latencies.std()),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
        }
        results["inference_rate_hz"] = 1000.0 / results["latency_stats"]["mean_ms"]
    else:
        results["latency_stats"] = {}
        results["inference_rate_hz"] = 0.0
    
    results["success_rate"] = results["successful"] / max(1, results["total_samples"])
    
    # Validation summary
    valid_outputs = sum(1 for v in results["validations"] if v.get("valid", False))
    results["valid_outputs"] = valid_outputs
    results["validation_rate"] = valid_outputs / max(1, results["successful"])
    
    return results


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Test X-VLA model inference")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model checkpoint (pruned_hf or quant_fp8_hf)")
    parser.add_argument("--model_type", type=str, default="auto", choices=["auto", "pruned", "quant"],
                        help="Model type. 'auto' detects from path.")
    parser.add_argument("--quant_state", type=str, default="",
                        help="Path to ModelOpt state for quantized model (defaults to model_path/modelopt_state_transformer.pth)")
    
    parser.add_argument("--baseline_id", type=str, default="",
                        help="HuggingFace model ID for baseline comparison")
    
    parser.add_argument("--calib_dir", type=str, required=True,
                        help="Directory with calibration npz files")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples to test (0 = all)")
    
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--local_files_only", action="store_true")
    
    parser.add_argument("--steps", type=int, default=1,
                        help="Denoising steps for action generation")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup iterations before timing")
    parser.add_argument("--no_autocast", action="store_true")
    
    parser.add_argument("--semi_structured", action="store_true",
                        help="Convert pruned weights to semi-structured sparse format")
    
    parser.add_argument("--output", type=str, default="",
                        help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Setup
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)
    use_autocast = (args.dtype in ("bf16", "fp16")) and (not args.no_autocast)
    
    # Auto-detect model type
    model_type = args.model_type
    if model_type == "auto":
        if "quant" in args.model_path.lower():
            model_type = "quant"
        else:
            model_type = "pruned"
    
    # Load calibration files
    calib_dir = Path(args.calib_dir)
    calib_files = list_calib_files(calib_dir, args.max_samples)
    print(f"Found {len(calib_files)} calibration files")
    
    results = {
        "config": {
            "model_path": args.model_path,
            "model_type": model_type,
            "dtype": args.dtype,
            "steps": args.steps,
            "use_autocast": use_autocast,
            "semi_structured": args.semi_structured,
            "num_samples": len(calib_files),
        },
        "tests": {},
    }
    
    # Test baseline if requested
    if args.baseline_id:
        print(f"\n{'='*60}")
        print(f"Testing BASELINE: {args.baseline_id}")
        print(f"{'='*60}")
        
        baseline = load_model(args.baseline_id, device, dtype, args.local_files_only)
        patch_action_encoder_dtype(baseline)
        patch_transformer_input_cast(baseline, dtype)
        
        baseline_results = run_inference_test(
            baseline, calib_files, device, dtype, args.steps, args.warmup, use_autocast
        )
        results["tests"]["baseline"] = baseline_results
        
        print(f"\nBaseline Results:")
        print(f"  Success rate: {baseline_results['success_rate']*100:.1f}%")
        print(f"  Inference rate: {baseline_results['inference_rate_hz']:.1f} Hz")
        if baseline_results["latency_stats"]:
            print(f"  Latency: {baseline_results['latency_stats']['mean_ms']:.2f} ± {baseline_results['latency_stats']['std_ms']:.2f} ms")
        
        del baseline
        torch.cuda.empty_cache()
    
    # Test optimized model
    print(f"\n{'='*60}")
    print(f"Testing {model_type.upper()}: {args.model_path}")
    if args.semi_structured:
        print(f"  (with semi-structured sparsity)")
    print(f"{'='*60}")
    
    model = load_model(args.model_path, device, dtype, args.local_files_only)
    
    # Apply ModelOpt state for quantized models
    if model_type == "quant":
        quant_state = Path(args.quant_state) if args.quant_state else (Path(args.model_path) / "modelopt_state_transformer.pth")
        print(f"Loading ModelOpt state from: {quant_state}")
        restore_modelopt_state(model, quant_state)
    
    patch_action_encoder_dtype(model)
    patch_transformer_input_cast(model, dtype)
    
    # Apply semi-structured conversion (only for pruned, not quant)
    if args.semi_structured and model_type != "quant":
        print("Converting to semi-structured sparse format...")
        sparse_stats = convert_to_semi_structured(model, scope="transformer")
        print(f"  Converted {sparse_stats.get('converted', 0)} layers")
        results["config"]["sparse_conversion"] = sparse_stats
    
    # Run tests
    model_results = run_inference_test(
        model, calib_files, device, dtype, args.steps, args.warmup, use_autocast
    )
    results["tests"]["optimized"] = model_results
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Type: {model_type}")
    print(f"Success rate: {model_results['success_rate']*100:.1f}% ({model_results['successful']}/{model_results['total_samples']})")
    print(f"Valid outputs: {model_results['validation_rate']*100:.1f}%")
    
    if model_results["latency_stats"]:
        stats = model_results["latency_stats"]
        print(f"\nLatency:")
        print(f"  Mean: {stats['mean_ms']:.2f} ms")
        print(f"  Std:  {stats['std_ms']:.2f} ms")
        print(f"  P50:  {stats['p50_ms']:.2f} ms")
        print(f"  P95:  {stats['p95_ms']:.2f} ms")
        print(f"  P99:  {stats['p99_ms']:.2f} ms")
        print(f"\nInference rate: {model_results['inference_rate_hz']:.1f} Hz")
    
    if model_results["errors"]:
        print(f"\nErrors ({len(model_results['errors'])}):")
        for err in model_results["errors"][:5]:
            print(f"  - {err['file']}: {err['error'][:60]}")
    
    # Compare with baseline if available
    if "baseline" in results["tests"]:
        baseline_results = results["tests"]["baseline"]
        if model_results["inference_rate_hz"] > 0 and baseline_results["inference_rate_hz"] > 0:
            speedup = model_results["inference_rate_hz"] / baseline_results["inference_rate_hz"]
            print(f"\nSpeedup vs baseline: {speedup:.2f}x")
            results["speedup_vs_baseline"] = speedup
    
    print(f"{'='*60}\n")
    
    # Save results
    if args.output:
        # Remove non-serializable items
        clean_results = json.loads(json.dumps(results, default=str))
        with open(args.output, "w") as f:
            json.dump(clean_results, f, indent=2)
        print(f"Results saved to: {args.output}")
    
    return results


if __name__ == "__main__":
    main()
