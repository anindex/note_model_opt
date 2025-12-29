# X-VLA Libero Optimization Bench, Pruning, FP8, TensorRT

This folder contains a small example pipeline to benchmark and improve inference latency for **2toINF/X-VLA-Libero**, with:
- PyTorch baseline timing
- "Pruned" HF checkpoint timing (with optional semi-structured sparse storage)
- ModelOpt FP8 transformer only timing (PyTorch)
- TensorRT engine build and transformer timing
- Optional "2:4 pruning at export time" to test TensorRT sparse tactics on a domain-specialized transformer

The main takeaway from this trial model opt run so far:
- TensorRT gives a strong speedup on the transformer portion.
- End to end latency is dominated by the VLM part, so overall gains are capped unless VLM is accelerated.
- 2:4 sparsity, even with 2:4 compliant weights, did not help on our current setup, it was slightly worse when sparse tactics were enabled.
- PyTorch semi-structured sparsity (2.1+) can reduce memory for pruned weights and enable faster sparse GEMM kernels on Ampere+ GPUs.

## Important note on scope and intent

This codebase is **an example of how the author built a model optimization pipeline for X-VLA** (benchmarking, export, TensorRT build, and a few pruning or quantization experiments). It is **not** presented as a canonical, standardized, or best practice workflow for X-VLA optimization. The intention is for purely visualizing the experimenter thought process.

Different projects, GPU architectures, TensorRT versions, and deployment constraints can lead to very different “right” pipelines. Treat the scripts here as a practical reference implementation and a starting point to adapt, rather than a one size fits all solution.

---

## Files

- `bench_xvla_variants.py`
  - Loads baseline, pruned, quant checkpoints.
  - Benchmarks VLM, policy, end to end (with a special path for transformer only variants).
  - Exports a domain-specialized transformer to ONNX.
  - Builds TensorRT engine, benchmarks transformer latency.

- `dump_xvla_calib_from_hf_libero.py`
  - Creates `.npz` calibration batches (input_ids, image_input, masks, etc).

- `xvla_trtllm_ptq_prune_build.py`
  - Pruning and PTQ build script with 2:4 pruning logic.
  - **Using `--prune_semi_structured`** to convert pruned weights to PyTorch's compressed sparse format.
  - Note: it prunes `nn.Linear`, but X-VLA uses `DomainAwareLinear` heavily, so pruning must happen after domain specialization to affect the real GEMM weights.

---

## Environment

Typical stack:
- Python 3.10
- `lerobot` with `libero` and `xvla`
- PyTorch 2.3 or newer (2.1+ required for semi-structured sparsity)
- TensorRT 10 or newer (the script supports TRT 8/9 and 10+ APIs)
- CUDA toolkit matching our PyTorch and TRT builds
- `nvidia-modelopt` for FP8 quant state
- Ampere+ GPU (SM80+: A100, H100, RTX 30xx/40xx/50xx) for semi-structured sparse GEMM acceleration

Important note:
- You will see `transformers==4.49.0` warning from `nvidia-modelopt`. That does not always break things, but mismatches can reduce stability.

---

## Quick start

### 1) Create calibration batches

```bash
python dump_xvla_calib_from_hf_libero.py \
  --tokenizer_repo 2toINF/X-VLA-Libero \
  --out_dir ./xvla_calib_libero_hf \
  --n_batches 32
````

This produces files like:

* `./xvla_calib_libero_hf/calib_00000.npz`

### 2) Build pruned checkpoint with semi-structured sparsity (memory-efficient)

```bash
python xvla_trtllm_ptq_prune_build.py \
  --model_id 2toINF/X-VLA-Libero \
  --calib_dir ./xvla_calib_libero_hf \
  --out_dir ./xvla_opt_out \
  --dtype bf16 \
  --do_prune --prune_scope transformer \
  --prune_semi_structured
```

This uses PyTorch's `torch.sparse.to_sparse_semi_structured()` to store pruned weights in compressed format and enables efficient sparse GEMM kernels.

### 3) Run the benchmark

```bash
python bench_xvla_variants.py \
  --baseline_id 2toINF/X-VLA-Libero \
  --pruned_ckpt ./xvla_opt_out/pruned_hf \
  --quant_ckpt ./xvla_opt_out/quant_fp8_hf \
  --calib_dir ./xvla_calib_libero_hf \
  --dtype fp16 \
  --vlm_precision policy \
  --vlm_autocast \
  --image_dtype fp16 \
  --iters 200 --warmup 20 --steps 1
```

### 4) Build and benchmark TensorRT transformer

Dense transformer engine:

```bash
python bench_xvla_variants.py \
  --baseline_id 2toINF/X-VLA-Libero \
  --calib_dir ./xvla_calib_libero_hf \
  --dtype fp16 --vlm_precision policy --vlm_autocast \
  --image_dtype fp16 \
  --iters 200 --warmup 20 --steps 1 \
  --trt --trt_fp16 \
  --trt_domain 0 \
  --trt_dir ./xvla_opt_out/trt \
  --trt_workspace_mb 4096 --trt_opset 18 \
  --trt_force_rebuild
```

2:4 pruning at export time, then build TRT engine:

```bash
python bench_xvla_variants.py \
  --baseline_id 2toINF/X-VLA-Libero \
  --calib_dir ./xvla_calib_libero_hf \
  --dtype fp16 --vlm_precision policy --vlm_autocast \
  --image_dtype fp16 \
  --iters 200 --warmup 20 --steps 1 \
  --trt --trt_fp16 \
  --trt_prune_2to4 \
  --trt_domain 0 \
  --trt_dir ./xvla_opt_out/trt \
  --trt_workspace_mb 4096 --trt_opset 18 \
  --trt_force_rebuild
```

Enable sparse tactic eligibility:

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
  --trt_workspace_mb 4096 --trt_opset 18 \
  --trt_force_rebuild
```

---

## How to interpret metrics

The script reports:

* `vlm_ms`: time spent in VLM forward (vision language encoder stage)
* `policy_ms`: time spent in policy forward (transformer plus small heads)
* `e2e_ms`: full `generate_actions` path, unless a variant is explicitly transformer only
* peak memory stats for each segment
* TensorRT `transformer_ms` for the exported domain-specialized transformer

Be careful about mixing measurements:

* The TRT `transformer_ms` is measured by running the TRT engine directly.
* You only get end to end gains if the TRT engine is actually used inside `generate_actions()`.

---

## Reasoning about the results you might get

### 1) Baseline is VLM dominated

In my run with RTX 5090, baseline numbers are stable across runs:

* `vlm_ms` ~ 11.8 ms
* `policy_ms` ~ 5.75 ms
* `e2e_ms` ~ 17.6 ms

VLM share is about:

* 11.8 / 17.6 ≈ 67%

This means even a huge policy speedup can only reduce end to end by so much, because VLM is the bottleneck.

### 2) "Pruned" PyTorch checkpoint does not help (without semi-structured)

our `pruned` variant is slightly worse than baseline and uses much more memory:

* `e2e_ms` increases from ~17.6 ms to ~17.7 to 17.8 ms
* peak memory jumps from ~1.9 GB to ~3.6 GB

This is expected if pruning is represented as dense weights with zeros, PyTorch does not automatically accelerate dense GEMMs with zero patterns, and additional wrappers or non-fused paths can increase memory.

 Use `--prune_semi_structured` to enable PyTorch's semi-structured sparsity:
- Stores 2:4 pruned weights in compressed format (50% memory reduction for weight storage)
- Enables efficient sparse GEMM kernels via CUTLASS on Ampere+ GPUs
- Requires PyTorch 2.1+ and SM80+ GPU (A100, H100, RTX 30xx/40xx/50xx)

Conclusion:

* The pruned HF checkpoint is not a PyTorch speed win by itself.

### 3) ModelOpt FP8 transformer only is slower in my current benchmark

The `quant_transformer_only` shows a much higher `policy_ms` than baseline:

* `policy_ms` ~ 13 to 13.6 ms, baseline is ~5.75 ms

This indicates the quantized path is not yet an efficient deployment path in the current measurement, common reasons:

* quant wrappers and dequant overhead on every layer
* kernel selection not optimal for our shapes
* missing fusion, missing TensorRT compilation, or mixed precision constraints

Conclusion:

* In this pipeline, FP8 transformer only is currently a regression, not a speedup.

### 4) TensorRT improves transformer latency significantly

The dense TRT transformer latency:

* `transformer_ms` ~ 2.466 ms

Compared to baseline policy time ~5.746 ms, that is:

* about 2.33x faster on the transformer chunk

If we could fully replace the transformer portion inside `generate_actions()` with the TRT engine, a first order bound is:

* new e2e ~ VLM + TRT transformer
* ~ 11.8 + 2.47 = 14.3 ms
* about 1.23x end to end speedup, about 19% lower latency

Conclusion:

* TRT is the biggest win you have so far, but it is limited by VLM.

### 5) 2:4 sparsity does not help, even with 2:4 compliance

We can run A/B on the same p24 ONNX. In my case:

* with `--trt_prune_2to4 --trt_sparse`: 2.4035 ms
* with `--trt_prune_2to4` only: 2.3588 ms

That means enabling sparse tactics was about:

* 1.9% slower

This suggests TensorRT either:

* does not choose sparse tensor core tactics for our exact GEMMs, or
* sparse tactics are available but not faster due to shape, fusion, or memory constraints

Also, earlier you were measuring on the default stream, which can add synchronizations and noise. Use the non-default stream `bench_trt_engine` so you can trust small deltas.

Conclusion:

* For our transformer configuration, 2:4 sparse tactics are not a win, and can be slightly worse.

---

## Potential next steps

### A) Make TRT gains real end to end

In the example, TRT transformer is measured separately. The next practical step is to integrate TRT engine execution into the policy path, so `policy_ms` in the main benchmark reflects TRT.

### B) Focus on VLM acceleration

Because VLM is about 67% of end to end latency, improving VLM is the path to larger gains:

* consider TensorRT for vision encoder blocks
* consider `torch.compile` for VLM if TRT is not feasible
* reduce image resolution or enforce static shapes where possible
* avoid dynamic control flow

### C) Do not chase 2:4 sparsity for this model shape yet

Given my measured results, 2:4 sparse tactics did not help. You can keep the pruning capability as an experiment, but it should not be the main optimization path unless you see a consistent improvement after fixing the stream timing and increasing iters.
