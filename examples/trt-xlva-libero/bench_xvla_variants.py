from __future__ import annotations

import argparse
import copy
import json
import inspect
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Tuple

import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel


# ----------------------------
# IO
# ----------------------------


def list_calib_files(calib_dir: Path, max_files: int) -> list[Path]:
    files = sorted(calib_dir.glob("calib_*.npz"))
    if not files:
        raise FileNotFoundError(f"No calib_*.npz found in {calib_dir}")
    return files[:max_files] if max_files and max_files > 0 else files


def load_npz_batch(
    p: Path,
    device: torch.device,
    proprio_dtype: torch.dtype,
    image_dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    arr = np.load(p)

    def req(name: str) -> np.ndarray:
        if name not in arr:
            raise KeyError(f"{p} missing key: {name}")
        return arr[name]

    input_ids = torch.from_numpy(req("input_ids")).to(device=device, dtype=torch.long)

    # Images are usually float32 in many vision stacks, but allowing fp16 helps memory/latency.
    image_input = torch.from_numpy(req("image_input")).to(device=device, dtype=image_dtype)

    image_mask = torch.from_numpy(req("image_mask")).to(device=device)
    if image_mask.dtype != torch.bool:
        image_mask = image_mask > 0

    domain_id = torch.from_numpy(req("domain_id")).to(device=device, dtype=torch.long)

    proprio = torch.from_numpy(req("proprio")).to(device=device, dtype=proprio_dtype)
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
# Model loading, dtype policy
# ----------------------------

def load_model(
    ckpt: str,
    device: torch.device,
    policy_dtype: torch.dtype,
    vlm_precision: str,
    local_files_only: bool,
) -> nn.Module:
    # HF compatibility across transformers versions
    try:
        m = AutoModel.from_pretrained(
            ckpt,
            trust_remote_code=True,
            dtype=policy_dtype,
            local_files_only=local_files_only,
        )
    except TypeError:
        m = AutoModel.from_pretrained(
            ckpt,
            trust_remote_code=True,
            torch_dtype=policy_dtype,
            local_files_only=local_files_only,
        )

    m.to(device)
    m.eval()

    # Policy module to policy dtype
    if hasattr(m, "transformer"):
        try:
            m.transformer.to(dtype=policy_dtype)
        except Exception:
            pass

    # VLM precision control
    # - fp32: move VLM to fp32 and do not autocast during forward_vlm
    # - policy: move VLM to policy dtype and allow autocast if enabled
    if hasattr(m, "vlm"):
        if vlm_precision == "fp32":
            try:
                m.vlm.to(dtype=torch.float32)
            except Exception:
                pass
        elif vlm_precision == "policy":
            try:
                m.vlm.to(dtype=policy_dtype)
            except Exception:
                pass

    return m


# ----------------------------
# Safety patches for X-VLA quirks
# ----------------------------

def _first_tensor_dtype(mod: nn.Module) -> torch.dtype | None:
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
    orig_forward = enc.forward

    def wrapped_forward(x: torch.Tensor, *args, **kwargs):
        target = _first_tensor_dtype(enc)
        if target is not None and torch.is_tensor(x) and x.dtype != target:
            x = x.to(dtype=target)
        return orig_forward(x, *args, **kwargs)

    enc.forward = wrapped_forward


def _normalize_gripper_idx(gi: Any, action_dim: int) -> list[int]:
    # Convert common types into a list of integer indices.
    if gi is None:
        return []
    if isinstance(gi, slice):
        return list(range(action_dim))[gi]
    if isinstance(gi, int):
        return [int(gi)]
    if torch.is_tensor(gi):
        gi = gi.detach().cpu().tolist()
    if isinstance(gi, (list, tuple)):
        out: list[int] = []
        for v in gi:
            try:
                out.append(int(v))
            except Exception:
                pass
        return out
    try:
        return [int(gi)]
    except Exception:
        return []


def sanitize_xvla_batch(model: nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Clamp token ids if vocab_size is present
    vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)
    if vocab_size > 0 and "input_ids" in batch:
        batch["input_ids"] = torch.clamp(batch["input_ids"], 0, vocab_size - 1)

    # Fix action/proprio dims and gripper_idx bounds
    if hasattr(model, "action_space") and "proprio" in batch:
        proprio = batch["proprio"]
        B = int(proprio.shape[0])

        # Prefer model-provided action dim, otherwise infer from proprio itself.
        action_dim = int(getattr(model.action_space, "dim_action", 0) or 0)
        if action_dim <= 0:
            action_dim = int(proprio.shape[-1])

        if action_dim <= 0:
            # Nothing we can do safely.
            return batch

        if int(proprio.shape[-1]) != action_dim:
            fixed = torch.zeros((B, action_dim), device=proprio.device, dtype=proprio.dtype)
            n = min(action_dim, int(proprio.shape[-1]))
            fixed[:, :n] = proprio[:, :n]
            batch["proprio"] = fixed

        gi = getattr(model.action_space, "gripper_idx", None)
        gi_list = _normalize_gripper_idx(gi, action_dim)

        safe = [i for i in gi_list if (-action_dim) <= i < action_dim]
        # Convert negative to positive
        safe = [(i if i >= 0 else action_dim + i) for i in safe]
        safe = [i for i in safe if 0 <= i < action_dim]

        if not safe:
            # Do not set anything dangerous, choose a conservative valid index.
            safe = [action_dim - 1]

        # Keep as list for advanced indexing semantics.
        try:
            model.action_space.gripper_idx = safe
        except Exception:
            pass

    return batch


def patch_action_space_preprocess_safe(model: nn.Module) -> None:
    """
    Extra guard: preprocess() may index action_m[..., gripper_idx].
    If gripper_idx is invalid, CUDA asserts and your process is poisoned.
    This wrapper clamps indices based on the runtime tensor shapes.
    """
    if not hasattr(model, "action_space"):
        return
    if not hasattr(model.action_space, "preprocess"):
        return

    orig = model.action_space.preprocess

    def wrapped_preprocess(proprio: torch.Tensor, x_t: torch.Tensor, *args, **kwargs):
        # Try to infer the dimension that gripper_idx will index.
        # Usually it is the last dim of an action-like tensor, which often matches proprio last dim.
        action_dim = int(getattr(model.action_space, "dim_action", 0) or 0)
        if action_dim <= 0 and torch.is_tensor(proprio) and proprio.ndim >= 2:
            action_dim = int(proprio.shape[-1])

        if action_dim > 0:
            gi = getattr(model.action_space, "gripper_idx", None)
            gi_list = _normalize_gripper_idx(gi, action_dim)
            safe = [i for i in gi_list if (-action_dim) <= i < action_dim]
            safe = [(i if i >= 0 else action_dim + i) for i in safe]
            safe = [i for i in safe if 0 <= i < action_dim]
            if not safe:
                safe = [action_dim - 1]
            try:
                model.action_space.gripper_idx = safe
            except Exception:
                pass

        return orig(proprio, x_t, *args, **kwargs)

    model.action_space.preprocess = wrapped_preprocess


# ----------------------------
# ModelOpt transformer-only restore
# ----------------------------

def restore_transformer_modelopt_state(model: nn.Module, state_path: Path) -> None:
    import modelopt.torch.opt as mto

    if not state_path.exists():
        raise FileNotFoundError(f"Missing transformer ModelOpt state: {state_path}")
    state = torch.load(state_path, map_location="cpu")
    if not hasattr(model, "transformer"):
        raise AttributeError("Model has no .transformer, cannot restore transformer-only state")
    mto.restore_from_modelopt_state(model.transformer, state)


# ----------------------------
# Benchmark helpers
# ----------------------------

@torch.inference_mode()
def bench_op(fn, warmup: int, iters: int) -> Tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    total_ms = float(start.elapsed_time(end))
    mean_ms = total_ms / float(iters)
    return mean_ms, total_ms


def _to_dtype(x: Any, dtype: torch.dtype) -> Any:
    if torch.is_tensor(x):
        return x.to(dtype=dtype)
    if isinstance(x, (list, tuple)):
        return type(x)(_to_dtype(v, dtype) for v in x)
    if isinstance(x, dict):
        return {k: _to_dtype(v, dtype) for k, v in x.items()}
    return x


@torch.inference_mode()
def bench_variant(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    steps: int,
    warmup: int,
    iters: int,
    use_autocast: bool,
    vlm_autocast: bool,
    do_e2e: bool = True,
) -> Dict[str, float]:
    device = next(model.parameters()).device
    assert device.type == "cuda", "This benchmark expects CUDA"

    batch = sanitize_xvla_batch(model, dict(batch))

    # Policy dtype comes from transformer if present
    policy_dtype = (
        next(model.transformer.parameters()).dtype
        if hasattr(model, "transformer")
        else next(model.parameters()).dtype
    )

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=policy_dtype)
        if use_autocast and policy_dtype in (torch.float16, torch.bfloat16)
        else nullcontext()
    )
    vlm_ctx = (
        torch.autocast(device_type="cuda", dtype=policy_dtype)
        if vlm_autocast and use_autocast and policy_dtype in (torch.float16, torch.bfloat16)
        else nullcontext()
    )

    # 1) VLM only
    torch.cuda.reset_peak_memory_stats()

    def vlm_fn():
        with vlm_ctx:
            _ = model.forward_vlm(batch["input_ids"], batch["image_input"], batch["image_mask"])

    vlm_ms, _ = bench_op(vlm_fn, warmup=warmup, iters=iters)
    vlm_peak = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

    # 2) Policy only (stub VLM output)
    with vlm_ctx:
        enc = model.forward_vlm(batch["input_ids"], batch["image_input"], batch["image_mask"])
    enc_for_policy = _to_dtype(enc, policy_dtype)

    orig_forward_vlm = model.forward_vlm
    model.forward_vlm = lambda *args, **kwargs: enc_for_policy
    try:
        torch.cuda.reset_peak_memory_stats()

        def policy_fn():
            with amp_ctx:
                _ = model.generate_actions(**batch, steps=steps)

        policy_ms, _ = bench_op(policy_fn, warmup=warmup, iters=iters)
        policy_peak = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    finally:
        model.forward_vlm = orig_forward_vlm

    # 3) E2E (optional)
    if do_e2e:
        torch.cuda.reset_peak_memory_stats()

        def e2e_fn():
            with amp_ctx:
                _ = model.generate_actions(**batch, steps=steps)

        e2e_ms, _ = bench_op(e2e_fn, warmup=warmup, iters=iters)
        e2e_peak = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    else:
        # For transformer-only variants (e.g., ModelOpt state restored only on transformer),
        # we treat E2E as "VLM + policy" while keeping the policy benchmark using a stubbed VLM output.
        e2e_ms = float(vlm_ms) + float(policy_ms)
        e2e_peak = float(max(vlm_peak, policy_peak))

    return {
        "vlm_ms": float(vlm_ms),
        "policy_ms": float(policy_ms),
        "e2e_ms": float(e2e_ms),
        "vlm_peak_mem_mb": float(vlm_peak),
        "policy_peak_mem_mb": float(policy_peak),
        "e2e_peak_mem_mb": float(e2e_peak),
        "iters": float(iters),
        "warmup": float(warmup),
        "steps": float(steps),
    }


# ----------------------------
# TensorRT export + build + run (transformer-only)
# ----------------------------

class _CaptureOnce:
    def __init__(self):
        self.args = None
        self.kwargs = None

    def __call__(self, args, kwargs):
        if self.args is None and self.kwargs is None:
            self.args = args
            self.kwargs = kwargs


@torch.inference_mode()
def capture_transformer_call(model: nn.Module, batch: Dict[str, torch.Tensor], steps: int) -> Tuple[tuple, dict]:
    """
    Captures the *actual* transformer.forward(args, kwargs) used inside generate_actions().
    This makes TRT export robust to signature order.
    """
    assert hasattr(model, "transformer"), "Model has no transformer"
    tf = model.transformer

    cap = _CaptureOnce()
    orig = tf.forward

    def wrapped_forward(*args, **kwargs):
        cap(args, kwargs)
        return orig(*args, **kwargs)

    tf.forward = wrapped_forward
    try:
        _ = model.generate_actions(**batch, steps=steps)
    finally:
        tf.forward = orig

    if cap.args is None:
        raise RuntimeError("Failed to capture transformer inputs, transformer.forward was not called")
    return cap.args, cap.kwargs or {}


def build_args_template_from_capture(tf: nn.Module, cap_args: tuple, cap_kwargs: dict) -> tuple:
    """Create a positional args template for transformer.forward from captured (args, kwargs).

    Many generate_actions() implementations call transformer.forward with keyword arguments only.
    Our TRT export path needs a stable positional template to:
      - identify the domain_id slot
      - enumerate tensor slots to become ONNX inputs
    """
    if cap_args and len(cap_args) > 0:
        return cap_args
    if not cap_kwargs:
        return cap_args

    # Prefer signature ordering when forward has named parameters
    try:
        sig = inspect.signature(tf.forward)
        params = [p for p in sig.parameters.values() if p.name != 'self']
        has_var = any(p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) for p in params)
        if (not has_var) and params:
            ordered = []
            for p in params:
                if p.name in cap_kwargs:
                    ordered.append(cap_kwargs[p.name])
                elif p.default is not inspect._empty:
                    ordered.append(p.default)
                else:
                    ordered.append(None)
            return tuple(ordered)
    except Exception:
        pass

    # Fallback: rely on keyword insertion order at call-site (Python preserves this)
    return tuple(cap_kwargs.values())


class FrozenDomainAwareLinear(nn.Module):
    """
    Keeps the same call signature as DomainAwareLinear: forward(x, domain_id),
    but uses fixed weights and ignores domain_id.
    """
    def __init__(self, weight_in_out: torch.Tensor, bias_out: torch.Tensor):
        super().__init__()
        # weight_in_out: [in, out]
        w = weight_in_out.t().contiguous()  # [out, in] for nn.Linear
        self.linear = nn.Linear(w.shape[1], w.shape[0], bias=True)
        with torch.no_grad():
            self.linear.weight.copy_(w)
            self.linear.bias.copy_(bias_out.view(-1))

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def specialize_transformer_domain(transformer: nn.Module, domain: int) -> nn.Module:
    """
    Replaces DomainAwareLinear modules by FrozenDomainAwareLinear for a fixed domain.
    This removes embedding gathers from the graph and helps export / TRT.
    """
    tf = copy.deepcopy(transformer).eval()
    tf = tf.cuda()

    # Lazy import, only used when TRT is requested
    import importlib

    # DomainAwareLinear class lives in the transformer's module
    mod = importlib.import_module(tf.__class__.__module__)
    DomainAwareLinear = getattr(mod, "DomainAwareLinear", None)
    if DomainAwareLinear is None:
        # Fall back: no specialization
        return tf

    device = next(tf.parameters()).device
    domain_id = torch.tensor([int(domain)], device=device, dtype=torch.long)

    def replace_in_module(parent: nn.Module):
        for name, child in list(parent.named_children()):
            if DomainAwareLinear is not None and isinstance(child, DomainAwareLinear):
                with torch.no_grad():
                    W = child.fc(domain_id).view(1, child.input_size, child.output_size)[0]  # [in, out]
                    b = child.bias(domain_id).view(1, child.output_size)[0]                 # [out]
                frozen = FrozenDomainAwareLinear(W, b).to(device=device, dtype=W.dtype)
                setattr(parent, name, frozen)
            else:
                replace_in_module(child)

    replace_in_module(tf)
    return tf


class TransformerTRTExport(nn.Module):
    """
    Wraps transformer.forward, removing domain_id from ONNX inputs.
    This expects the transformer.forward signature:
      (domain_id, vlm_features, aux_visual_inputs, action_with_noise, proprio, t)
    but we do not assume order. We use captured args order.
    """
    def __init__(self, tf: nn.Module, args_template: tuple, domain: int):
        super().__init__()
        self.tf = tf
        self.args_template = args_template
        self.domain = int(domain)

    
    def forward(self, *inputs):
        # Fill a mutable list matching captured args, replacing domain_id with a constant.
        tmpl = list(self.args_template)
    
        # Determine device
        if len(inputs) > 0 and hasattr(inputs[0], 'device'):
            device = inputs[0].device
        else:
            device = next(self.tf.parameters()).device
    
        # Determine batch size (B)
        B = None
        for x in inputs:
            if hasattr(x, 'ndim') and x.ndim >= 1:
                B = int(x.shape[0])
                break
        if B is None:
            for v in tmpl:
                if hasattr(v, 'ndim') and v.ndim >= 1:
                    B = int(v.shape[0])
                    break
        if B is None:
            B = 1
    
        const_domain = torch.full((B,), self.domain, device=device, dtype=torch.long)
    
        # Identify domain slot: first 1D LongTensor in template, and count remaining tensor slots.
        replaced_domain = False
        tensor_slots = []
        for i, v in enumerate(tmpl):
            if torch.is_tensor(v):
                if (not replaced_domain) and v.dtype == torch.long and v.ndim == 1:
                    replaced_domain = True
                    continue
                tensor_slots.append(i)
    
        if len(inputs) != len(tensor_slots):
            raise RuntimeError(
                f"TRT export input count mismatch: got {len(inputs)} tensors but expected {len(tensor_slots)}. "
                f"(Transformer may have been called with kwargs only; ensure args_template is built from kwargs.)"
            )
    
        # Replace domain slot and fill remaining tensor slots from inputs, in order.
        replaced_domain = False
        in_it = iter(inputs)
        for i, v in enumerate(tmpl):
            if (not replaced_domain) and torch.is_tensor(v) and v.dtype == torch.long and v.ndim == 1:
                tmpl[i] = const_domain
                replaced_domain = True
            elif torch.is_tensor(v):
                tmpl[i] = next(in_it)
            else:
                pass
    
        return self.tf(*tmpl)
    

def export_onnx_dynamo(
    model: nn.Module,
    example_inputs: tuple,
    onnx_path: Path,
    opset: int,
) -> None:
    """Export to ONNX.

    We try the dynamo exporter first (uses torch.export), then fall back to the legacy exporter.
    The fallback is important because some PyTorch builds fail inside torch.export for certain graphs.
    """
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    kwargs = dict(opset_version=int(opset))

    # 1) Try dynamo exporter
    dynamo_ok = False
    try:
        torch.onnx.export(
            model,
            example_inputs,
            str(onnx_path),
            dynamo=True,
            **kwargs,
        )
        dynamo_ok = True
    except TypeError:
        # Older API: no dynamo flag
        dynamo_ok = False
    except Exception as e:
        print(f"[trt] torch.onnx dynamo export failed: {type(e).__name__}: {e}. Falling back to legacy exporter...", flush=True)
        dynamo_ok = False

    # 2) Legacy exporter fallback
    if not dynamo_ok:
        try:
            torch.onnx.export(
                model,
                example_inputs,
                str(onnx_path),
                dynamo=False,
                **kwargs,
            )
        except TypeError:
            # Even older API: no dynamo kwarg at all
            torch.onnx.export(
                model,
                example_inputs,
                str(onnx_path),
                **kwargs,
            )

    try:
        import onnx
        from onnx import TensorProto

        m = onnx.load(str(onnx_path))
        needs_external = False
        for init in m.graph.initializer:
            if init.data_location == TensorProto.EXTERNAL:
                needs_external = True
                break

        if needs_external:
            data_name = onnx_path.name + ".data"
            onnx.save_model(
                m,
                str(onnx_path),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=data_name,
                size_threshold=1024,
            )
    except Exception:
        pass


def build_trt_engine(
    onnx_path: Path,
    engine_path: Path,
    workspace_mb: int,
    fp16: bool,
    sparse: bool,
    verbose: bool,
) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Important: when the ONNX model uses external data (a sidecar ".onnx.data" file),
    # TensorRT must parse from *file* so it can resolve the external weights.
    # NOTE: if `onnx_path` is relative, passing it after `chdir` will break. We therefore:
    #   1) resolve to an absolute path
    #   2) chdir to the directory containing the ONNX
    #   3) call parse_from_file with the *basename*.
    onnx_abs = onnx_path.resolve()
    old_cwd = None
    try:
        old_cwd = os.getcwd()
        os.chdir(str(onnx_abs.parent))
        ok = parser.parse_from_file(onnx_abs.name)
    except AttributeError:
        # Older TRT: no parse_from_file, fall back to bytes (only works if ONNX has no external data).
        ok = parser.parse(onnx_abs.read_bytes())
    finally:
        if old_cwd is not None:
            try:
                os.chdir(old_cwd)
            except Exception:
                pass

    if not ok:
        msgs = []
        for i in range(parser.num_errors):
            msgs.append(str(parser.get_error(i)))
        raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(msgs))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_mb) * 1024 * 1024)

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Enable 2:4 structured sparsity acceleration when the network contains compatible weights.
    # Not all TRT builds expose this flag, so guard it.
    if sparse:
        try:
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        except Exception:
            pass

    # Create a fixed-shape optimization profile (min=opt=max for each input)
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = tuple(int(d) for d in inp.shape)
        profile.set_shape(inp.name, shape, shape, shape)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build_serialized_network returned None")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))


def prune_2to4_lastdim(w: torch.Tensor) -> torch.Tensor:
    """Apply 2:4 structured sparsity along the output (first) dimension.
    
    For nn.Linear.weight with shape [out_features, in_features], TensorRT's
    sparse tensor cores expect 2:4 sparsity along the M (output) dimension.
    We group every 4 output channels and zero the 2 smallest magnitudes.
    """
    if (not torch.is_tensor(w)) or w.ndim != 2 or (w.shape[0] % 4) != 0:
        return w
    # Reshape to [out//4, 4, in] to group along output dimension
    out_dim, in_dim = w.shape
    w4 = w.reshape(out_dim // 4, 4, in_dim)  # [out//4, 4, in]
    # Find 2 smallest magnitudes along the group-of-4 dimension (dim=1)
    _, idx = torch.topk(w4.abs(), k=2, dim=1, largest=False)  # [out//4, 2, in]
    w4p = w4.clone()
    w4p.scatter_(1, idx, 0)
    return w4p.reshape_as(w)


def fraction_groups_2to4(w: torch.Tensor) -> float:
    """Fraction of 4-tuples having exactly 2 zeros along output dimension.
    
    For nn.Linear.weight [out, in], checks sparsity pattern along the output
    dimension to verify TensorRT 2:4 sparse tensor core compatibility.
    """
    if (not torch.is_tensor(w)) or w.ndim != 2 or (w.shape[0] % 4) != 0:
        return 0.0
    out_dim, in_dim = w.shape
    w4 = w.reshape(out_dim // 4, 4, in_dim)  # [out//4, 4, in]
    z = (w4 == 0).sum(dim=1)  # [out//4, in] - count zeros in each group of 4
    return float((z == 2).float().mean().item())


@torch.inference_mode()
def bench_trt_engine(engine_path: Path, inputs: tuple, warmup: int, iters: int) -> float:
    """
    Run a TensorRT engine and return average latency (ms).

    Supports both the legacy "binding" API (TRT 8/9) and the newer I/O tensor API (TRT 10+).
    Uses a non-default CUDA stream to avoid extra synchronizations in enqueueV3().
    """
    import tensorrt as trt
    import torch

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context")

    # Some TRT builds expose explicit optimization profiles
    if hasattr(context, "active_optimization_profile"):
        try:
            context.active_optimization_profile = 0
        except Exception:
            pass

    # Non-default CUDA stream
    trt_stream = torch.cuda.Stream()
    stream_handle = trt_stream.cuda_stream

    # -----------------------
    # TRT 10+: I/O tensor API
    # -----------------------
    if hasattr(engine, "num_io_tensors") and hasattr(engine, "get_tensor_name"):
        n_io = int(engine.num_io_tensors)
        io_names = [engine.get_tensor_name(i) for i in range(n_io)]
        input_names = [n for n in io_names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        output_names = [n for n in io_names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

        if len(inputs) != len(input_names):
            raise RuntimeError(f"TRT engine expects {len(input_names)} inputs but got {len(inputs)}")

        # Set input shapes and addresses
        for name, x in zip(input_names, inputs):
            if not torch.is_tensor(x):
                raise TypeError(f"Input for {name} must be a torch.Tensor")
            x = x.contiguous()

            if hasattr(context, "set_input_shape"):
                context.set_input_shape(name, tuple(int(d) for d in x.shape))
            elif hasattr(context, "set_tensor_shape"):
                context.set_tensor_shape(name, tuple(int(d) for d in x.shape))
            else:
                raise RuntimeError("TensorRT context does not support setting input shapes")

            context.set_tensor_address(name, int(x.data_ptr()))

        # Allocate outputs after shapes are known
        output_tensors: list[torch.Tensor] = []
        for name in output_names:
            if hasattr(context, "get_tensor_shape"):
                shp = tuple(int(d) for d in context.get_tensor_shape(name))
            else:
                shp = tuple(int(d) for d in engine.get_tensor_shape(name))

            dt = engine.get_tensor_dtype(name)
            if dt == trt.DataType.HALF:
                torch_dt = torch.float16
            elif dt == trt.DataType.FLOAT:
                torch_dt = torch.float32
            elif hasattr(trt.DataType, "BF16") and dt == trt.DataType.BF16:
                torch_dt = torch.bfloat16
            else:
                torch_dt = torch.float32

            y = torch.empty(shp, device="cuda", dtype=torch_dt)
            output_tensors.append(y)
            context.set_tensor_address(name, int(y.data_ptr()))

        def run_once() -> None:
            if hasattr(context, "execute_async_v3"):
                ok = context.execute_async_v3(stream_handle=stream_handle)
            else:
                ok = context.execute_async_v2(bindings=[], stream_handle=stream_handle)
            if not ok:
                raise RuntimeError("TensorRT execution failed")

    # -----------------------
    # TRT 8/9: binding API
    # -----------------------
    else:
        bindings: list[int] = []
        output_tensors: list[torch.Tensor] = []
        input_idx = 0

        n_bindings = int(engine.num_bindings)
        for bi in range(n_bindings):
            if engine.binding_is_input(bi):
                if input_idx >= len(inputs):
                    raise RuntimeError(f"TRT expects more inputs than provided, binding {bi} is input")
                x = inputs[input_idx]
                input_idx += 1
                if not torch.is_tensor(x):
                    raise TypeError(f"Binding {bi} input must be a torch.Tensor")
                x = x.contiguous()
                context.set_binding_shape(bi, tuple(int(d) for d in x.shape))
                bindings.append(int(x.data_ptr()))
            else:
                out_shape = tuple(int(d) for d in context.get_binding_shape(bi))
                dt = engine.get_binding_dtype(bi)
                if dt == trt.DataType.HALF:
                    torch_dt = torch.float16
                elif dt == trt.DataType.FLOAT:
                    torch_dt = torch.float32
                elif hasattr(trt.DataType, "BF16") and dt == trt.DataType.BF16:
                    torch_dt = torch.bfloat16
                else:
                    torch_dt = torch.float32
                y = torch.empty(out_shape, device="cuda", dtype=torch_dt)
                output_tensors.append(y)
                bindings.append(int(y.data_ptr()))

        if input_idx != len(inputs):
            raise RuntimeError(f"TRT binding API consumed {input_idx} inputs but got {len(inputs)}")

        def run_once() -> None:
            ok = context.execute_async_v2(bindings=bindings, stream_handle=stream_handle)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v2 failed")

    # Warmup and timing on TRT stream
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(trt_stream):
        for _ in range(int(warmup)):
            run_once()
    trt_stream.synchronize()

    with torch.cuda.stream(trt_stream):
        start.record()
        for _ in range(int(iters)):
            run_once()
        end.record()
    trt_stream.synchronize()

    return float(start.elapsed_time(end)) / float(iters)

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--baseline_id", type=str, default="2toINF/X-VLA-Libero")
    ap.add_argument("--pruned_ckpt", type=str, default="./xvla_opt_out/pruned_hf")
    ap.add_argument("--quant_ckpt", type=str, default="./xvla_opt_out/quant_fp8_hf")
    ap.add_argument("--quant_state", type=str, default="", help="Defaults to <quant_ckpt>/modelopt_state_transformer.pth")

    ap.add_argument("--calib_dir", type=str, required=True)
    ap.add_argument("--calib_max_files", type=int, default=1)

    ap.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--vlm_precision", type=str, default="fp32", choices=["fp32", "policy"])
    ap.add_argument(
        "--image_dtype",
        type=str,
        default="fp32",
        choices=["fp16", "fp32"],
        help="Dtype for image_input loaded from calib npz. fp16 reduces memory and may improve speed.",
    )
    ap.add_argument("--local_files_only", action="store_true")

    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--steps", type=int, default=1)

    ap.add_argument("--no_autocast", action="store_true")
    ap.add_argument("--vlm_autocast", action="store_true", help="Only meaningful when --vlm_precision policy")
    ap.add_argument("--quant_do_e2e", action="store_true", help="Also benchmark quant checkpoint end-to-end (runs VLM inside generate_actions). Default is transformer-only E2E = vlm_ms + policy_ms.")


    # TensorRT (transformer-only)
    ap.add_argument("--trt", action="store_true")
    ap.add_argument("--trt_domain", type=int, default=0)
    ap.add_argument("--trt_dir", type=str, default="./xvla_opt_out/trt")
    ap.add_argument("--trt_opset", type=int, default=18)
    ap.add_argument("--trt_workspace_mb", type=int, default=4096)
    ap.add_argument("--trt_fp16", action="store_true", help="Build TRT engine with FP16 flag")
    ap.add_argument("--trt_sparse", action="store_true", help="Enable TensorRT sparse weights (2:4) if supported")
    ap.add_argument("--trt_prune_2to4", action="store_true", help="Apply 2:4 pruning to the domain-specialized transformer before ONNX export (so TRT can use sparse kernels)")
    ap.add_argument("--trt_force_rebuild", action="store_true")
    ap.add_argument("--trt_verbose", action="store_true")

    args = ap.parse_args()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    policy_dtype = dtype_map[args.dtype]
    image_dtype = dtype_map[args.image_dtype]

    use_autocast = (args.dtype in ("bf16", "fp16")) and (not args.no_autocast)
    vlm_autocast = bool(args.vlm_autocast) and (args.vlm_precision == "policy")

    device = torch.device("cuda")
    calib_files = list_calib_files(Path(args.calib_dir), max_files=max(1, args.calib_max_files))
    batch = load_npz_batch(calib_files[0], device=device, proprio_dtype=policy_dtype, image_dtype=image_dtype)

    out: Dict[str, Any] = {
        "env": {
            "dtype": args.dtype,
            "use_autocast": use_autocast,
            "vlm_precision": args.vlm_precision,
            "vlm_autocast": vlm_autocast,
            "image_dtype": args.image_dtype,
            "calib_used": str(calib_files[0]),
        }
    }

    # Baseline
    baseline = load_model(args.baseline_id, device=device, policy_dtype=policy_dtype, vlm_precision=args.vlm_precision, local_files_only=args.local_files_only)
    patch_action_encoder_dtype(baseline)
    patch_action_space_preprocess_safe(baseline)
    r_base = bench_variant(baseline, batch, steps=args.steps, warmup=args.warmup, iters=args.iters, use_autocast=use_autocast, vlm_autocast=vlm_autocast)
    r_base["ckpt"] = args.baseline_id
    out["baseline"] = r_base

    # Pruned
    pruned = load_model(args.pruned_ckpt, device=device, policy_dtype=policy_dtype, vlm_precision=args.vlm_precision, local_files_only=args.local_files_only)
    patch_action_encoder_dtype(pruned)
    patch_action_space_preprocess_safe(pruned)
    r_pruned = bench_variant(pruned, batch, steps=args.steps, warmup=args.warmup, iters=args.iters, use_autocast=use_autocast, vlm_autocast=vlm_autocast)
    r_pruned["ckpt"] = args.pruned_ckpt
    out["pruned"] = r_pruned

    # Quant (transformer-only state)
    quant = load_model(args.quant_ckpt, device=device, policy_dtype=policy_dtype, vlm_precision=args.vlm_precision, local_files_only=args.local_files_only)
    q_state = Path(args.quant_state) if args.quant_state else (Path(args.quant_ckpt) / "modelopt_state_transformer.pth")
    restore_transformer_modelopt_state(quant, q_state)
    patch_action_encoder_dtype(quant)
    patch_action_space_preprocess_safe(quant)
    r_quant = bench_variant(quant, batch, steps=args.steps, warmup=args.warmup, iters=args.iters, use_autocast=use_autocast, vlm_autocast=vlm_autocast, do_e2e=args.quant_do_e2e)
    r_quant["ckpt"] = args.quant_ckpt
    r_quant["state"] = str(q_state)
    out["quant_transformer_only"] = r_quant

    out["speedup_vs_baseline"] = {
        "pruned_e2e_x": float(r_base["e2e_ms"] / r_pruned["e2e_ms"]),
        "quant_e2e_x": float(r_base["e2e_ms"] / r_quant["e2e_ms"]),
        "pruned_policy_x": float(r_base["policy_ms"] / r_pruned["policy_ms"]),
        "quant_policy_x": float(r_base["policy_ms"] / r_quant["policy_ms"]),
    }

    # TensorRT path: build and benchmark transformer-only
    if args.trt:
        try:
            # Use baseline for TRT by default
            batch_trt = sanitize_xvla_batch(baseline, dict(batch))
            # IMPORTANT: capture must run with transformer input dtypes matching transformer weights.
            # The default X-VLA VLM may emit fp32 features, which can cause dtype mismatch errors
            # (e.g., float input + half bias). We therefore stub forward_vlm with a cached,
            # policy-dtype output, similar to bench_variant().
            tf_weight_dtype = next(baseline.transformer.parameters()).dtype
            with torch.no_grad():
                enc = baseline.forward_vlm(batch_trt["input_ids"], batch_trt["image_input"], batch_trt["image_mask"])
            enc_for_policy = _to_dtype(enc, tf_weight_dtype)

            orig_forward_vlm = baseline.forward_vlm
            baseline.forward_vlm = lambda *a, **k: enc_for_policy
            try:
                with torch.no_grad():
                    tf_args, tf_kwargs = capture_transformer_call(baseline, batch_trt, steps=args.steps)
            finally:
                baseline.forward_vlm = orig_forward_vlm

            # Convert captured kwargs into positional tensors if your transformer was called via kwargs.
            # Most X-VLA calls positional, so we focus on tf_args.
            # Prepare tensor-only args (exclude domain_id) matching template
            args_template = build_args_template_from_capture(baseline.transformer, tf_args, tf_kwargs)

            # Create a specialized transformer for the given domain
            frozen_tf = specialize_transformer_domain(baseline.transformer, domain=int(args.trt_domain)).eval().cuda()

            # Optionally apply 2:4 pruning to the *domain-specialized* transformer before export.
            # This is the correct place for X-VLA because DomainAwareLinear gets frozen into nn.Linear here.
            if args.trt_prune_2to4:
                pruned = 0
                checked = 0
                for lname, lm in frozen_tf.named_modules():
                    if isinstance(lm, torch.nn.Linear) and getattr(lm, "weight", None) is not None:
                        w = lm.weight.data
                        if w.ndim == 2 and (w.shape[1] % 4 == 0):
                            lm.weight.data.copy_(prune_2to4_lastdim(w))
                            pruned += 1
                            if args.trt_verbose and checked < 3:
                                frac = fraction_groups_2to4(lm.weight.data)
                                print(f"[trt] 2:4 check {lname}: {frac*100:.1f}% groups")
                                checked += 1
                if args.trt_verbose:
                    print(f"[trt] Applied 2:4 pruning to {pruned} nn.Linear layers in frozen_tf")

            # Build export wrapper, domain_id becomes internal constant
            export_mod = TransformerTRTExport(frozen_tf, args_template=args_template, domain=int(args.trt_domain)).eval().cuda()

            # Prepare example tensor inputs by taking the tensor slots from template, skipping domain_id slot
            tensor_inputs = []
            replaced_domain = False
            for v in args_template:
                if (not replaced_domain) and torch.is_tensor(v) and v.dtype == torch.long and v.ndim == 1:
                    replaced_domain = True
                    continue
                if torch.is_tensor(v):
                    tensor_inputs.append(v)

            if len(tensor_inputs) == 0:
                raise RuntimeError("Captured transformer inputs had no tensor arguments besides domain_id; cannot export to TRT")

            # Cast float inputs to fp16 or fp32 depending on TRT build choice
            if args.trt_fp16:
                tensor_inputs = [x.to(dtype=torch.float16) if (torch.is_tensor(x) and x.is_floating_point()) else x for x in tensor_inputs]
            else:
                tensor_inputs = [x.to(dtype=torch.float32) if (torch.is_tensor(x) and x.is_floating_point()) else x for x in tensor_inputs]

            tensor_inputs = tuple(x.contiguous() for x in tensor_inputs)

            trt_dir = Path(args.trt_dir)
            suffix = ""
            if args.trt_prune_2to4:
                suffix += "_p24"
            onnx_path = trt_dir / f"transformer_domain{int(args.trt_domain)}_{'fp16' if args.trt_fp16 else 'fp32'}{suffix}.onnx"
            engine_path = trt_dir / f"transformer_domain{int(args.trt_domain)}_{'fp16' if args.trt_fp16 else 'fp32'}{suffix}.plan"

            if args.trt_force_rebuild or (not engine_path.exists()):
                export_onnx_dynamo(export_mod, tensor_inputs, onnx_path=onnx_path, opset=int(args.trt_opset))
                build_trt_engine(
                    onnx_path=onnx_path,
                    engine_path=engine_path,
                    workspace_mb=int(args.trt_workspace_mb),
                    fp16=bool(args.trt_fp16),
                    sparse=bool(args.trt_sparse),
                    verbose=bool(args.trt_verbose),
                )

            trt_ms = bench_trt_engine(engine_path, tensor_inputs, warmup=max(10, args.warmup // 2), iters=args.iters)
            out["trt_transformer_only"] = {
                "domain": int(args.trt_domain),
                "fp16": bool(args.trt_fp16),
                "onnx": str(onnx_path),
                "engine": str(engine_path),
                "transformer_ms": float(trt_ms),
            }
        except Exception as e:
            out["trt_transformer_only_error"] = str(e)

    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
