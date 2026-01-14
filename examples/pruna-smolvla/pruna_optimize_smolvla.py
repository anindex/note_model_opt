"""
Optimize SmolVLA with Pruna (TorchAO quantization + torch.compile).

Install:
  pip install lerobot "lerobot[smolvla]" pruna==0.2.10

Examples:
  # INT8 weight-only (what worked for me)
  python pruna_optimize_smolvla.py --quantizer torchao --torchao-quant-type int8wo

  # INT8 dynamic
  python pruna_optimize_smolvla.py --quantizer torchao --torchao-quant-type int8dq

  # Half precision
  python pruna_optimize_smolvla.py --quantizer half

What works (Pruna 0.2.10):
  - TorchAO int8wo, int8dq, half, torch.compile

What doesn't:
  - int4wo (tensor shape error), HQQ, pruning ("not compatible")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


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
    """Count trainable params."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pruna optimization for SmolVLA (TorchAO + torch.compile)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Model options
    ap.add_argument(
        "--model-id",
        default="lerobot/smolvla_base",
        help="HuggingFace model ID (default: lerobot/smolvla_base)",
    )
    ap.add_argument(
        "--out-dir",
        default="./out/smolvla_pruna_smashed",
        help="Output directory for optimized model",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="Device: cpu|cuda|mps (default: auto-detect)",
    )
    
    # === Quantization options ===
    # Note: HQQ and structured pruning are NOT compatible with SmolVLA in Pruna 0.2.10
    ap.add_argument(
        "--quantizer",
        choices=["torchao", "half", "none"],
        default="torchao",
        help="Quantization backend (default: torchao). Note: HQQ is not compatible with SmolVLA.",
    )
    
    # TorchAO-specific
    ap.add_argument(
        "--torchao-quant-type",
        default=None,
        help="TorchAO quant type: int8wo, int8dq (recommended). "
             "Note: int4wo/int4dq fail with SmolVLA. (default: int8wo on CUDA, int8dq on CPU)",
    )
    ap.add_argument(
        "--torchao-excluded-modules",
        default="norm+embedding",
        help="TorchAO excluded modules: none, norm, embedding, norm+embedding "
             "(default: norm+embedding)",
    )
    
    # === Compilation options ===
    ap.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip torch.compile (faster iteration, less warmup)",
    )
    
    args = ap.parse_args()
    
    device = pick_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[pruna-smolvla] Device: {device}")
    print(f"[pruna-smolvla] Output: {out_dir.resolve()}")
    
    # === Load SmolVLA policy ===
    print(f"[pruna-smolvla] Loading model: {args.model_id}")
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError:
        # Fallback for older lerobot versions
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(args.model_id)
    policy = policy.to(device).eval()
    
    baseline_params = count_parameters(policy)
    print(f"[pruna-smolvla] Baseline parameters: {baseline_params:,}")
    
    # === Configure Pruna SmashConfig ===
    from pruna import SmashConfig, smash
    
    cfg = SmashConfig(device=device)
    
    # --- 1. Quantization ---
    if args.quantizer != "none":
        print(f"[pruna-smolvla] Configuring quantizer: {args.quantizer}")
        cfg["quantizer"] = args.quantizer
        
        if args.quantizer == "torchao":
            cfg["torchao_excluded_modules"] = args.torchao_excluded_modules
            # Note: int4wo doesn't work with SmolVLA, use int8wo as default
            cfg["torchao_quant_type"] = (
                args.torchao_quant_type
                if args.torchao_quant_type is not None
                else ("int8wo" if device == "cuda" else "int8dq")
            )
            print(f"[pruna-smolvla]   torchao_quant_type: {cfg['torchao_quant_type']}")
            print(f"[pruna-smolvla]   torchao_excluded_modules: {cfg['torchao_excluded_modules']}")
        
        elif args.quantizer == "half":
            print("[pruna-smolvla]   Using FP16 half precision")
    
    # --- 2. Compilation ---
    if not args.no_compile:
        print("[pruna-smolvla] Configuring compiler: torch_compile")
        cfg["compiler"] = "torch_compile"
    
    # === Run Pruna smash ===
    print("\n[pruna-smolvla] Running Pruna smash...")
    print("[pruna-smolvla] Optimization stack: ", end="")
    stack = []
    if args.quantizer != "none":
        if args.quantizer == "torchao":
            stack.append(f"quantize({args.quantizer}, {cfg['torchao_quant_type']})")
        else:
            stack.append(f"quantize({args.quantizer})")
    if not args.no_compile:
        stack.append("compile(torch_compile)")
    print(" → ".join(stack) if stack else "(none)")
    
    optimized = smash(model=policy, smash_config=cfg)
    
    # === Save optimized model ===
    saved = False
    
    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Try different save methods (API varies across Pruna versions)
    if hasattr(optimized, "save_pretrained"):
        try:
            optimized.save_pretrained(str(out_dir))
            saved = True
        except Exception as e:
            print(f"[pruna-smolvla] Warning: save_pretrained failed: {e}")
    
    if not saved and hasattr(optimized, "save_model"):
        try:
            optimized.save_model(str(out_dir))
            saved = True
        except Exception as e:
            print(f"[pruna-smolvla] Warning: save_model failed: {e}")
    
    # Save config to the output directory
    # Note: Pruna's save_to_json may append filename, so pass directory
    if hasattr(cfg, "save_to_json"):
        try:
            cfg.save_to_json(str(out_dir))
        except Exception as e:
            print(f"[pruna-smolvla] Warning: save_to_json failed: {e}")
    
    # === Summary ===
    print("\n" + "=" * 60)
    print("[pruna-smolvla] Optimization complete!")
    print("=" * 60)
    print(f"  Model ID:       {args.model_id}")
    print(f"  Device:         {device}")
    print(f"  Quantizer:      {args.quantizer}" + (
        f" ({cfg.get('torchao_quant_type', '')})" if args.quantizer == "torchao" else ""
    ))
    print(f"  Compilation:    {'enabled' if not args.no_compile else 'disabled'}")
    print(f"  Baseline params: {baseline_params:,}")
    print(f"  Output saved:   {saved}")
    print(f"  Output dir:     {out_dir.resolve()}")
    print("=" * 60)
    
    print("\nNext steps:")
    print("  1. Benchmark: python bench_smolvla_pruna.py --baseline-id lerobot/smolvla_base --smashed-dir", out_dir)
    print("  2. Evaluate on your robot task to verify policy quality")


if __name__ == "__main__":
    main()
