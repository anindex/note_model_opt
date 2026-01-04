# A collection of notes on model deployment

This repo is **not** a polished library or a finished survey.  
It is simply a place where I collect working notes, small experiments, and mental maps around:

## Notes

- [MLSoC.md](./notes/MLSoC.md): tour of the non-CUDA SoC landscape: Vulkan/Kompute, OpenCL, MLX, ARM Compute Library/ARM NN, oneDNN, TVM, IREE, ncnn, ExecuTorch/LiteRT, plus vendor stacks (Qualcomm QNN, TI TIDL, NXP eIQ, NNAPI).
- [OptimizingModels.md](./notes/OptimizingModels.md): model and edge-level optimization mainstream pathways: quantization (PTQ/QAT, FP8/INT4), pruning/sparsity, distillation, low-rank adapters, and deployment toolchains (TensorRT ModelOpt, Intel INC/OpenVINO, TorchAO, ONNX Runtime, TFLite, TVM, ncnn).
- [ModelOptDeepDive.md](./notes/ModelOptDeepDive.md): a deeper dive into the **model-side optimization stack** for foundation models, focusing less on edge runtime layers and more on the end-to-end pipeline that makes big models efficient in practice, including weight quantization (INT4/INT8), KV-cache compression, sparsity and sparse kernels, decoding-time acceleration (speculative decoding, multi-head decoding), and high-impact serving kernels and engines (FlashAttention, vLLM, TensorRT-LLM), with pointers to codebases for each component.

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
- PyTorch semi-structured sparsity has overhead; real acceleration requires TensorRT
- TensorRT provides the most practical speedup path

Scripts: calibration data generation, pruning + quantization build, benchmarking, inference testing.

### [X-VLA Typecast Benchmark](./examples/typecast-xvla-libero/)

Benchmarking harness for X-VLA on LIBERO simulation with inference optimizations:

- AMP autocast (bf16/fp16)
- TF32 tensor cores
- Flash SDP attention
- Policy-only latency measurement (isolates model speed from simulation)

---

If you are okay with half-baked ideas, TODOs, and rough edges, you might find something useful here.
If you see something obviously wrong or missing, PRs are all very welcome.
