# A collection of notes on model deployment & optimization

This repo is **not** a polished library or a finished survey.  
It is a place where I collect working notes, small experiments, and mental maps around *making models run fast* (training, inference, and serving).

---

## Notes

- [MLSoC.md](./notes/MLSoC.md): ML training + inference on **SoCs (non-CUDA landscape)** — Vulkan/Kompute, Metal/Core ML/MLX, **LiteRT** (ex‑TFLite), ExecuTorch, ONNX Runtime, TVM/IREE, ncnn; kernel libs (XNNPACK, oneDNN, Arm Compute Library/Arm NN, CMSIS‑NN, KleidiAI); plus vendor stacks (Qualcomm QNN, TI TIDL, NXP eIQ, etc.) and NNAPI deprecation/migration notes.
- [VLAonSocs.md](./notes/VLAonSocs.md): focusing on **VLA deployment on SoCs** (robotics/embodied) - end-to-end VLA vs VLM+policy, quant/prune/distill, caching/scheduling tricks, and device lanes (Core ML/ANE, QNN, TensorRT, RKNN, LiteRT/ORT, MLC/llama.cpp) + profiling/packaging checklist.
- [OptimizingModels.md](./notes/OptimizingModels.md): a two-layer view of optimization:
  - **Model-level**: PTQ/QAT quantization (INT8/INT4/FP8/NVFP4), pruning/sparsity, distillation, low-rank/adapters
  - **Deployment + serving**: NVIDIA **Model Optimizer (ModelOpt)** + TensorRT‑LLM, OpenVINO/NNCF, TorchAO/PT2E, ONNX Runtime, LiteRT, TVM, ncnn; plus “systems wins” (continuous batching, paged KV, chunked prefill, prefix cache) and kernel ecosystems (FlashAttention / FlashInfer).
- [ModelOptDeepDive.md](./notes/ModelOptDeepDive.md): deeper dive into the “what actually moves latency/throughput” stack for foundation models — weight quant (INT4/INT8/FP8/NVFP4), KV-cache compression/quantization, sparse kernels (incl. 2:4), speculative + multi-token decoding, and high-impact serving kernels/engines (vLLM, TensorRT‑LLM, TGI, SGLang).
- [PrunaAI.md](./notes/Pruna.md): dedicated note on **Pruna.ai** as an end-to-end model optimization layer (compression search, PTQ/QAT, pruning, distillation, exports, eval loops), with caveats + “what model families it tends to crush”, and comparisons to similar frameworks (ModelOpt, OpenVINO/NNCF, INC, Olive, TorchAO, Optimum, LLM Compressor, etc.).

---

## Examples

Working pipelines that apply techniques from the notes above:

### [X-VLA Libero Optimization](./examples/trt-xlva-libero/)

End-to-end optimization pipeline for [2toINF/X-VLA-Libero](https://huggingface.co/2toINF/X-VLA-Libero) X-VLA model:

| Technique | Result |
|-----------|--------|
| 2:4 Structured Pruning | 1.02x (dense storage in PyTorch) |
| ModelOpt FP8 Quantization | 0.78x (fake quant overhead) |
| **TensorRT FP16 + Sparse** | **2.11x policy speedup** |

**Key findings:**
- VLM dominates E2E latency (~71%), capping overall gains to ~1.17x
- PyTorch semi-structured sparsity has overhead; real acceleration requires TensorRT kernels
- TensorRT provides the most practical speedup path (for NVIDIA targets)

Scripts: calibration data generation, pruning + quantization build, benchmarking, inference testing.

### [X-VLA Typecast Benchmark](./examples/typecast-xvla-libero/)

Benchmarking harness for X-VLA on LIBERO simulation with inference optimizations:

- AMP autocast (bf16/fp16)
- TF32 tensor cores
- Flash SDP attention
- Policy-only latency measurement (isolates model speed from simulation)

### [Pruna SmolVLA Optimization](./examples/pruna-smolvla/)

Practical workflow for optimizing SmolVLA (LeRobot VLA policy) using **Pruna**:

| Method | Status | Notes |
|--------|--------|-------|
| TorchAO int8wo | ✅ Works | Weight-only INT8, ~36% memory savings |
| TorchAO int8dq | ✅ Works | Dynamic INT8 quantization |
| Half precision | ✅ Works | FP16 conversion |
| torch.compile | ✅ Works | Graph compilation |
| HQQ / Pruning | ❌ Fails | Not compatible with SmolVLA in Pruna 0.2.10 |

**Key findings:**
- INT8 quantization provides reliable memory reduction (~36%) with minimal quality loss
- INT4 and HQQ are not compatible with SmolVLA's architecture in current Pruna version
- Structured pruning not supported for SmolVLA

Scripts: calibration data generation, optimization pipeline, benchmarking.

---

If you are okay with half-baked ideas, TODOs, and rough edges, you might find something useful here.  
If you see something obviously wrong or missing, PRs are all very welcome.
