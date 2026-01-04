# X-VLA Libero Optimization: Pruning, Quantization & TensorRT

This folder contains an optimization pipeline for **2toINF/X-VLA-Libero** with:
- 2:4 structured pruning with optional semi-structured sparse format
- FP8/INT8 quantization via NVIDIA ModelOpt
- TensorRT engine build for accelerated inference
- Comprehensive benchmarking and inference testing

## Key Results Summary

| Variant | Policy Latency | Speedup | Notes |
|---------|---------------|---------|-------|
| Baseline (PyTorch FP16) | 17.09 ms | 1.00x | Reference |
| Pruned (dense, PyTorch) | 16.77 ms | 1.02x | 2:4 sparsity pattern, dense storage |
| Quantized FP8 (PyTorch) | 21.88 ms | 0.78x | ModelOpt fake quantization overhead |
| **TensorRT FP16 + 2:4 Sparse** | **8.11 ms** | **2.11x** | Best option for deployment |

**Key Takeaways:**
- **TensorRT provides 2.11x speedup** on the transformer/policy portion
- End-to-end latency is VLM-dominated (~71% of time), capping overall gains to ~1.17x E2E
- PyTorch semi-structured sparsity has overhead; real sparse acceleration requires TensorRT
- FP8 quantization in PyTorch adds overhead; benefits require TensorRT INT8/FP8 deployment

## Important Note on Scope and Intent

This codebase demonstrates **one approach** to X-VLA optimization. It is not a canonical or one-size-fits-all solution. Different GPUs, TensorRT versions, and deployment constraints may require different pipelines. The intention is to visualize the experimenter's thought process.

---

## Files

| File | Description |
|------|-------------|
| `bench_xvla_variants.py` | Benchmark baseline, pruned, quantized models + TensorRT export |
| `dump_xvla_calib_from_hf_libero.py` | Generate calibration batches from LIBERO dataset |
| `xvla_trtllm_ptq_prune_build.py` | Build optimized checkpoints (pruning + quantization) |
| `test_xvla_inference.py` | Test inference success rate and throughput |

---

## Technical Considerations

1. **2:4 Pruning Dimension**: Pruning is applied along the K (in_features) dimension, which is what TensorRT and cuSPARSELt expect for sparse tensor cores.

2. **Semi-Structured Sparsity Limitations**: PyTorch's `SparseSemiStructuredTensor` is prototype-stage and:
   - Has overhead that can make inference slower than dense
   - Is incompatible with ModelOpt FP8/INT8 quantization ops
   - Real speedup requires TensorRT export

3. **DomainAwareLinear**: X-VLA uses `DomainAwareLinear` which must be specialized to a single domain before pruning affects the actual GEMM weights.

---

## Environment Setup

**Requirements:**
- Python 3.10+
- PyTorch 2.3+ (2.1+ for semi-structured sparsity)
- CUDA 12.x with matching toolkit
- TensorRT 10+ (supports TRT 8/9 APIs too)
- Ampere+ GPU (SM80+: A100, H100, RTX 30xx/40xx/50xx)

**Installation:**
```bash
pip install nvidia-modelopt transformers==4.49.0
pip install onnxruntime-gpu onnxscript tensorrt
pip install fastapi json_numpy uvicorn timm
```

---

## Quick Start

### 1) Generate Calibration Data

```bash
python dump_xvla_calib_from_hf_libero.py \
  --tokenizer_repo 2toINF/X-VLA-Libero \
  --out_dir ./xvla_calib_libero_hf \
  --n_batches 16
```

### 2) Build Optimized Checkpoints

**Pruned + Quantized (recommended for TensorRT):**
```bash
python xvla_trtllm_ptq_prune_build.py \
  --model_id 2toINF/X-VLA-Libero \
  --calib_dir ./xvla_calib_libero_hf \
  --out_dir ./xvla_opt_out \
  --dtype bf16 \
  --do_prune --prune_scope transformer \
  --do_quant --quant fp8 --ptq_scope transformer \
  --export_mode transformer_state \
  --calib_max_files 16 --denoise_steps 1
```

**With semi-structured sparsity (experimental):**
```bash
python xvla_trtllm_ptq_prune_build.py \
  --model_id 2toINF/X-VLA-Libero \
  --calib_dir ./xvla_calib_libero_hf \
  --out_dir ./xvla_opt_out \
  --dtype bf16 \
  --do_prune --prune_scope transformer \
  --prune_semi_structured \
  --do_quant --quant fp8 --ptq_scope transformer \
  --export_mode transformer_state \
  --calib_max_files 16 --denoise_steps 1
```

### 3) Benchmark All Variants

**PyTorch benchmarks only:**
```bash
python bench_xvla_variants.py \
  --baseline_id 2toINF/X-VLA-Libero \
  --pruned_ckpt ./xvla_opt_out/pruned_hf \
  --quant_ckpt ./xvla_opt_out/quant_fp8_hf \
  --calib_dir ./xvla_calib_libero_hf \
  --dtype fp16 --vlm_precision policy --vlm_autocast \
  --image_dtype fp16 \
  --iters 200 --warmup 20 --steps 1
```

**With TensorRT (recommended):**
```bash
python bench_xvla_variants.py \
  --baseline_id 2toINF/X-VLA-Libero \
  --calib_dir ./xvla_calib_libero_hf \
  --dtype fp16 --vlm_precision policy --vlm_autocast \
  --image_dtype fp16 \
  --iters 200 --warmup 20 --steps 1 \
  --trt --trt_fp16 \
  --trt_prune_2to4 \
  --trt_sparse \
  --trt_domain 0 \
  --trt_dir ./xvla_opt_out/trt \
  --trt_workspace_mb 4096 --trt_opset 18
```

### 4) Test Inference Success Rate

**Test pruned model:**
```bash
python test_xvla_inference.py \
  --model_path ./xvla_opt_out/pruned_hf \
  --baseline_id 2toINF/X-VLA-Libero \
  --calib_dir ./xvla_calib_libero_hf \
  --dtype bf16 \
  --output results.json
```

**Test quantized model:**
```bash
python test_xvla_inference.py \
  --model_path ./xvla_opt_out/quant_fp8_hf \
  --model_type quant \
  --calib_dir ./xvla_calib_libero_hf \
  --dtype bf16 \
  --output results_quant.json
```

---

## Benchmark Results

### PyTorch Performance

```
============================================================
PYTORCH PERFORMANCE SUMMARY
============================================================
Baseline policy:        17.09 ms
Pruned (dense) policy:  16.77 ms (1.02x)
Quantized policy:       21.88 ms (0.78x)
============================================================
```

### TensorRT Performance

```
============================================================
TENSORRT RESULTS
============================================================
TRT Transformer:        8.11 ms (2.11x vs baseline policy)
TRT E2E (estimated):    49.02 ms (1.17x vs baseline E2E)
  (VLM: 40.91 ms + TRT: 8.11 ms)
============================================================
```

### Inference Success Rate (Pruned Model)

```
============================================================
SUMMARY
============================================================
Model: ./xvla_opt_out/pruned_hf
Type: pruned
Success rate: 100.0% (16/16)
Valid outputs: 100.0%

Latency:
  Mean: 58.05 ms
  Std:  7.44 ms
  P50:  55.05 ms
  P95:  71.30 ms
  P99:  78.60 ms

Inference rate: 17.2 Hz
Speedup vs baseline: 0.99x
============================================================
```

---

## Understanding the Results

### 1) VLM Dominates End-to-End Latency

| Component | Time | Share |
|-----------|------|-------|
| VLM (Florence2) | 40.91 ms | ~71% |
| Policy (Transformer) | 17.09 ms | ~29% |
| **Total E2E** | **57.57 ms** | 100% |

Even a 2x policy speedup only yields ~1.17x E2E improvement because VLM is the bottleneck.

### 2) PyTorch Pruning Shows Minimal Gains

The pruned model (dense storage) is only 1.02x faster:
- Dense GEMMs with zero patterns are not automatically accelerated
- Sparse pattern benefits require specialized kernels (TensorRT or cuSPARSELt)

### 3) ModelOpt FP8 Has Overhead in PyTorch

The quantized model is 0.78x (slower):
- Fake quantization wrappers add overhead
- Real FP8 benefits require hardware tensor cores via TensorRT

### 4) TensorRT Provides Real Speedup

TensorRT transformer is **2.11x faster**:
- Graph optimization and kernel fusion
- FP16 tensor core utilization
- 2:4 sparse tactics can be enabled (though gains vary by GPU)

---

## Metrics Explained

| Metric | Description |
|--------|-------------|
| `vlm_ms` | Vision-Language Model forward time |
| `policy_ms` | Transformer/policy forward time |
| `e2e_ms` | Full `generate_actions()` latency |
| `transformer_ms` | TensorRT engine execution time |
| `*_peak_mem_mb` | Peak GPU memory during segment |

**Important:** TRT `transformer_ms` is measured by running the engine directly. To get real E2E gains, the TRT engine must be integrated into `generate_actions()`.

---

## Recommendations

### For Deployment
1. **Use TensorRT** for the transformer portion (2.11x speedup)
2. **Keep VLM in FP16** with autocast for stability
3. **Skip PyTorch semi-structured** unless you have specific memory constraints

### For Further Optimization
1. **Accelerate VLM** - This is 71% of latency
   - TensorRT for vision encoder
   - `torch.compile` for Florence2
   - Reduce image resolution if acceptable
2. **Integrate TRT into E2E path** - Replace PyTorch transformer with TRT engine in `generate_actions()`
3. **Profile with Nsight** - Identify remaining bottlenecks

### What Does Not Help (Currently)
- **PyTorch semi-structured sparsity**: Adds overhead, incompatible with quantization
- **ModelOpt FP8 in PyTorch**: Fake quantization overhead outweighs benefits
- **2:4 sparse tactics in TRT**: May not help depending on GPU and matrix shapes

---

## Output Directory Structure

```
xvla_opt_out/
├── pruned_hf/                    # Pruned HuggingFace + checkpoint
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files...
├── quant_fp8_hf/                 # Quantized checkpoint + ModelOpt state
│   ├── config.json
│   ├── model.safetensors
│   ├── modelopt_state_transformer.pth
│   └── tokenizer files...
├── trt/                          # TensorRT artifacts
│   ├── transformer_domain0_fp16_p24.onnx
│   └── transformer_domain0_fp16_p24.plan
└── report.json                   # Build report
```

---

## Troubleshooting

### "Float and BFloat16" dtype mismatch
The VLM outputs FP32 features but transformer expects BF16/FP16. The scripts handle this automatically via dtype casting hooks.

### Semi-structured conversion fails
Ensure weights have exact 2:4 sparsity pattern (2 zeros per 4 elements along K dimension) and are in FP16/BF16.

### TRT build fails
- Check CUDA/TensorRT version compatibility
- Increase `--trt_workspace_mb` if OOM
- Use `--trt_verbose` for detailed logs

### ModelOpt quantization errors
- Ensure `transformers==4.49.0` for X-VLA compatibility
- Use `weights_only=False` when loading ModelOpt states
