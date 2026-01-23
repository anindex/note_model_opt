# Model Optimization in 2025–2026: A Practical Survey

**An T. Le, Hanoi, Dec 2025 (revised Jan 2026)**

Foundation models keep getting larger, context windows keep getting longer, and deployment constraints keep getting tighter. In practice, “model optimization” is no longer one trick: it’s a stack of techniques that touch **weights**, **activations**, **KV cache**, **kernels**, **runtime scheduling**, and **distributed training**.

This note focuses on what *actually* changes memory / latency / throughput in real systems, and links to **working code** + **official docs** where possible.

---

## The optimization stack (what you actually pay for)

When you profile an LLM service, you usually see a handful of dominant cost centers:

- **Weights** (VRAM footprint; bandwidth during decoding)
- **KV cache** (VRAM footprint; bandwidth; fragmentation under multi-tenant serving; long-context blowups)
- **Attention and GEMM kernels** (fusion, IO efficiency, precision support)
- **Scheduling** (continuous batching, prefill vs decode balance, cache reuse / prefix caching)
- **Training memory** (optimizer states, gradients, sharding, precision)

Everything below maps to one or more of those bottlenecks.

---

## 1. Quantization (largest memory win; usually best first move)

### 1.1 Weight-only PTQ (keep activations high precision; quantize weights to 4-bit)

**Reality check:** weight-only INT4 helps most when **decode is bandwidth-bound** *and* you have **good INT4 kernels** (packing format matters).

**Algorithms / toolkits**
- **GPTQ (research baseline):** [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq)
- **GPTQModel (practical/maintained GPTQ family):** [ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel)  
  (AutoGPTQ is archived/unmaintained; GPTQModel is the common maintained path now.)
- **AWQ:** [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)  
  Common “accurate INT4” choice when you can calibrate quickly.
-- **EXL2 quantization** [Turboderp] (https://github.com/turboderp-org/exllamav3)
  (ExLlamaV3 aims to address this with the EXL3 format, which is a streamlined variant of QTIP from Cornell RelaxML)

**Docs that matter (format + runtime support)**
- **Transformers quantization guide (GPTQ/AWQ/bitsandbytes):** https://huggingface.co/docs/transformers/en/main_classes/quantization
- **TGI quantization support matrix (what kernels it actually uses):** https://huggingface.co/docs/text-generation-inference/en/conceptual/quantization
- **vLLM quantization docs (what loads & runs):** https://docs.vllm.ai/en/latest/features/quantization/
- **GGUF / llama.cpp quant formats (local CPU/GPU inference):**  
  - llama.cpp repo: https://github.com/ggml-org/llama.cpp  
  - quantize tool docs (K-quants, etc): https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md

**Discussion: when weight-only wins**
- You are **decode-bound** (memory bandwidth limited).
- You want a **drop-in** path without engineering activation quantization.
- Your serving engine supports the exact **packing + kernel** combination (often the hidden make-or-break detail).

---

### 1.2 W8A8 (INT8 weights + INT8 activations): higher peak performance, higher sensitivity

- **SmoothQuant:** https://github.com/mit-han-lab/smoothquant  
  Good bridge for vendor stacks where INT8 kernels are excellent.

**Discussion: why W8A8 is tricky**
- Activations have **outliers** that dominate quantization error.
- Calibration data + per-layer behavior matters more than weight-only.
- Speedups require **excellent INT8 kernels/fusions** in your stack.

---

### 1.3 FP8 / FP4 formats (where modern GPU stacks are going)

If you serve on recent GPUs, FP8 (and increasingly 4-bit float formats like NVFP4) are “mainline” in vendor stacks:

- **TensorRT supported quantized types (INT4/INT8/FP8):**  
  https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html
- **TensorRT-LLM quantization examples (incl. fp8_kv_cache / nvfp4):**  
  https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/quantization/README.md
- **NVIDIA Model Optimizer (ModelOpt) docs:** https://nvidia.github.io/Model-Optimizer/  
  Repo: https://github.com/NVIDIA/Model-Optimizer

Rule of thumb:
- FP8 often shines for **prefill throughput** (compute-heavy).
- INT4 weight-only often shines for **decode throughput** (bandwidth-heavy), assuming kernels are good.

---

### 1.4 Practical k-bit infrastructure (what people actually use)

- **bitsandbytes (8-bit + 4-bit; training + inference workflows):**  
  Repo: https://github.com/bitsandbytes-foundation/bitsandbytes  
  HF docs: https://huggingface.co/docs/transformers/en/quantization/bitsandbytes
- **TorchAO / pytorch/ao (PyTorch-native quantization + sparsity; integrates with torch.compile + FSDP2):**  
  Docs: https://docs.pytorch.org/ao/stable/  
  Repo: https://github.com/pytorch/ao
- **compressed-tensors (unified checkpoint format for quant + sparsity variants):**  
  HF docs: https://huggingface.co/docs/transformers/en/quantization/compressed_tensors
- **vector-quantize-pytorch (VQ building blocks for training):**
  Repo: https://github.com/lucidrains/vector-quantize-pytorch

---

### 1.5 4-bit fine-tuning without full fine-tune cost (QLoRA)

- **QLoRA (original):** https://github.com/artidoro/qlora
- **PEFT (practical LoRA/QLoRA tooling):** https://github.com/huggingface/peft  
  PEFT quantization guide: https://huggingface.co/docs/peft/en/developer_guides/quantization

**Discussion: what you really get with QLoRA**
- You optimize the **training iteration loop** more than final inference speed.
- Deployment can be “serve adapters” vs “merge adapters” — different ops/latency tradeoffs.

---

### 1.6 Ultra-low-bit PTQ (2–4 bits): pushing compression harder

Useful when INT4 is not enough, or for edge/local constraints (but kernels decide if it’s *fast* or just *small*).

- **AutoRound:** https://github.com/intel/auto-round
- **AQLM:** https://github.com/Vahe1994/AQLM
- **HQQ:** https://github.com/dropbox/hqq
- **HAWQ:** https://github.com/Zhen-Dong/HAWQ
  - *Hessian AWare Quantization (mixed-precision sensitivity analysis).*

**Discussion: reality of 2–3 bits**
- Accuracy is workload-dependent → validate on real prompts.
- Kernel support often decides if you get speedups or only memory savings.

---

## 2. KV cache optimization (the hidden bottleneck for long context + concurrency)

If you serve long prompts or many concurrent users, KV cache can dominate GPU memory and cause sharp perf cliffs via fragmentation.

### 2.1 “In-engine” KV cache quantization (practical, shipping)

- **vLLM Quantized KV Cache (FP8/INT8):** https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
- **NVIDIA NVFP4 KV cache (Blackwell-oriented):**  
  https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/

These are usually higher-ROI than research KV compression if you’re already on vLLM / TRT-LLM.

### 2.2 Research KV cache compression / pruning methods

- **ZipCache:** https://github.com/ThisisBillhe/ZipCache
- **KVQuant:** https://github.com/SqueezeAILab/KVQuant
- **KIVI (2-bit KV quantization):** https://github.com/jy-yuan/KIVI
- **SnapKV:** https://github.com/FasterDecoding/SnapKV  
  (repo notes compatibility constraints; treat as research code that may need porting)

**Discussion: when KV cache work matters most**
- Long context chat, heavy RAG prompts, tool traces, multi-turn sessions.
- High concurrency where KV cache memory is the capacity limiter.
- Systems needing predictable latency (cache pressure → sudden cliffs).

---

## 3. Sparsity and pruning (powerful on paper; conditional in practice)

Pruning can reduce compute and memory, but real speedups depend heavily on **sparsity structure** and **kernel support**.

### 3.1 One-shot pruning for LLMs

- **SparseGPT:** https://github.com/IST-DASLab/sparsegpt

### 3.2 Semi-structured 2:4 sparsity (hardware-aligned path)

- **Sparse-Marlin (INT4 + 2:4 kernel work):** https://github.com/IST-DASLab/Sparse-Marlin
- **cuSPARSELt (2:4 sparse GEMM library):** https://docs.nvidia.com/cuda/cusparselt/
- **TorchAO sparsity docs (semi-structured, block, etc.):** https://docs.pytorch.org/ao/stable/sparsity.html

**Discussion: when sparsity is worth it**
- You can enforce a hardware-friendly pattern (often **2:4**).
- You control kernels + deployment environment (not just the checkpoint).
- You have benchmarks showing sparse kernels beat dense low-bit for your shapes + batch regime.

---

## 4. Decoding acceleration (reduce the number of expensive target-model steps)

Autoregressive decoding is sequential; methods that reduce target forward passes can yield big speedups *if integrated into the serving stack well*.

### 4.1 Speculative decoding (draft model → verify with target)

- **vLLM speculative decoding docs:** https://docs.vllm.ai/en/latest/features/spec_decode/
- **TensorRT-LLM speculative decoding docs:** https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html
- **SGLang speculative decoding ecosystem:**  
  - Serving framework: https://github.com/sgl-project/sglang  
  - Training speculative models: https://github.com/sgl-project/SpecForge

(If you just want a minimal reference implementation: https://github.com/romsto/Speculative-Decoding)

### 4.2 Multi-token / multi-head decoding (Medusa-family)

- **Medusa:** https://github.com/FasterDecoding/Medusa
- **EAGLE (speculative-style but feature-extrapolation):** https://github.com/SafeAILab/EAGLE
- **SpecInfer (tree-based speculative serving concept):** https://arxiv.org/html/2305.09781v4

**Discussion: when decoding tricks pay off**
- You can maintain high acceptance rates (draft model quality matters).
- Your serving engine’s integration is efficient (overhead can cancel gains).

---

## 5. Kernels and serving engines (where “real throughput” often comes from)

You can have a well-quantized model and still get poor performance if the runtime can’t batch efficiently or manage KV cache well.

### 5.1 Attention kernels

- **FlashAttention (FA2 baseline):** https://github.com/Dao-AILab/flash-attention
- **FlashAttention-3 (Hopper-focused techniques; blog):** https://tridao.me/blog/2024/flash3/
- **FlashInfer (serving-focused kernel library + docs):**  
  Repo: https://github.com/flashinfer-ai/flashinfer  
  Docs: https://docs.flashinfer.ai/

### 5.2 Serving engines

- **vLLM:** https://github.com/vllm-project/vllm (docs: https://docs.vllm.ai/)
- **TensorRT-LLM:** https://nvidia.github.io/TensorRT-LLM/ (repo: https://github.com/NVIDIA/TensorRT-LLM)
- **Hugging Face TGI:** https://github.com/huggingface/text-generation-inference (docs: https://huggingface.co/docs/text-generation-inference)
- **SGLang:** https://github.com/sgl-project/sglang

### 5.3 Scheduling-level wins (often bigger than “one more quant trick”)

If your GPU isn’t saturated, these can dominate:

- **Automatic prefix caching (reuse KV for shared prefixes):**  
  vLLM docs: https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/
- **Chunked prefill (interleave big prefills with decode to reduce tail latency):**  
  vLLM docs: https://docs.vllm.ai/en/stable/configuration/optimization/#chunked-prefill
- **Disaggregated prefill/decode (separate compute-bound prefill from bandwidth-bound decode):**  
  - vLLM: https://docs.vllm.ai/en/latest/features/disagg_prefill/  
  - TensorRT-LLM blog: https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html

(If you’re CPU-heavy: OpenVINO Model Server also documents continuous batching + paged attention ideas:  
https://docs.openvino.ai/2024/ovms_demos_continuous_batching.html)

### 5.4 “Production glue” (optional but common): Triton

If you want a standard serving layer + batching + observability, Triton often sits above the engine:
- **TensorRT-LLM backend:** https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/README.html
- **vLLM backend:** https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/vllm_backend/README.html

### 5.5 Profiling Tools

- **CUTracer:** https://github.com/facebookresearch/CUTracer
  - *A lightweight CUDA kernel tracer for functional correctness debugging and performance profiling.*
- **ExCa:** https://github.com/facebookresearch/exca
  - *Execution Cascade Analysis: A tool for analyzing CUDA execution graphs.*
- **Kineto:** https://github.com/pytorch/kineto
  - *Holistic Performance Analysis of AI Workloads on GPUs.*
- **cuGraph:** https://github.com/rapidsai/cugraph
  - *RAPIDS Graph Analytics Library (profiling/analysis utilities).*
- **NVBit:** https://github.com/NVlabs/NVBit
  - *NVIDIA Binary Instrumentation Tool for dynamic analysis of CUDA kernels.*
- **CUPTI:** https://developer.nvidia.com/cupti
  - *The CUDA Profiling Tools Interface (foundational profiling API).*
- **ONNX Runtime Profiling Tools:** https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html
  - *Integrated performance profiling for ONNX Runtime (CPU/GPU/NPU).*
- **Qualcomm Profiler:** https://www.qualcomm.com/developer/software/qualcomm-profiler
  - *System and kernel-level profiling for Snapdragon SoCs (Hexagon NPU/Adreno GPU).*

### 5.6 Vision Kernels (Preprocessing)
- **CV-CUDA:** https://github.com/CVCUDA/CV-CUDA
  - *High-performance GPU-accelerated computer vision (image processing/augmentation) kernels.*

### 5.7 GEMM Optimization
- **CUDA-GEMM-Optimization:** https://github.com/leimao/CUDA-GEMM-Optimization
  - *Detailed tutorials and implementations for optimizing FP32/FP16 matmul on CUDA.*

### 5.8 Transformer Layer Optimizations
- **PyTorch BetterTransformer:** https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/
  - *PyTorch-native fast path for TransformerEncoder/Decoder (sparsity + fusion).*
- **Qualcomm Efficient Transformers:** https://github.com/quic/efficient-transformers
  - *Optimized Transformer primitives (often edge/mobile focused).*

**Practical serving mindset**
1. Optimize utilization first: batching, KV management, prefill vs decode scheduling.
2. Then optimize math: quantization, kernels, decoding shortcuts.
3. Validate quality under your exact sampler settings and prompt distribution.

---

## 6. Training-scale optimization (making big runs feasible)

Large training and large-scale fine-tuning are often memory-limited by optimizer state + gradients + replicas.

### 6.1 ZeRO-style sharding and offload

- **DeepSpeed:** https://github.com/deepspeedai/DeepSpeed  
  ZeRO docs: https://deepspeed.readthedocs.io/en/latest/zero3.html  
  Examples: https://github.com/deepspeedai/DeepSpeedExamples

### 6.2 PyTorch-native sharding (FSDP + FSDP2)

- **PyTorch FSDP docs:** https://docs.pytorch.org/docs/stable/fsdp.html
- **FSDP2 (`fully_shard`) docs:** https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
- **TorchTitan (clean-room reference for PyTorch-native scaling techniques):** https://github.com/pytorch/torchtitan

### 6.3 Parallelism reference stack (if you go deep)

- **Megatron-LM:** https://github.com/NVIDIA/Megatron-LM

### 6.4 FP8 building blocks

- **TransformerEngine:** https://github.com/NVIDIA/TransformerEngine

---

## 7. Diffusion and generative vision models (fewer steps beats faster steps)

For diffusion-style models, denoising step count often dominates latency.

- **Latent Consistency Models (LCM repo):** https://github.com/luosiallen/latent-consistency-model
- **Diffusers LCM inference guide (LCM + LCM-LoRA):** https://huggingface.co/docs/diffusers/en/using-diffusers/inference_with_lcm
- **torch.compile + diffusers performance guide:** https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/
- **Stability AI generative-models hub:** https://github.com/Stability-AI/generative-models

---

## 8. Putting it together (high-signal stacks)

### A. High-throughput LLM serving (GPU)

- Engine: vLLM or TensorRT-LLM (or TGI if you want HF-managed production ergonomics)
- Kernel: FlashAttention/FlashInfer where supported
- Quant: INT4 weight-only (AWQ/GPTQModel) for decode-heavy workloads; FP8 for prefill-heavy workloads
- KV: enable FP8/NVFP4 KV cache if long-context or high concurrency is the limiter
- Decode accel: speculative decoding (draft model) if acceptance rates are high

### B. Local / edge-ish constraints (single node, cost-sensitive)

- Quantize to a runtime-friendly format (GGUF / llama.cpp; or a supported GPTQ/AWQ format for your engine).
- Prefer distillation (smaller model) before extreme quant, if quality matters.

### C. Training memory wall

- Start with DeepSpeed ZeRO or PyTorch FSDP2
- Add activation checkpointing + optimizer tricks
- Use FP8 carefully (TransformerEngine / supported stacks)

---

### Quick checklist (what to profile first)

- Are you **prefill-bound** (compute) or **decode-bound** (bandwidth)?
- Is **KV cache** the capacity limiter (context/concurrency)?
- Are you actually using the best **attention/GEMM kernels** your stack supports?
- Is batching (continuous/dynamic) enabled and stable under your traffic shape?

