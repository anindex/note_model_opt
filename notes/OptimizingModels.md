# Optimizing Models: A Train Of Thought

**An T. Le, Hanoi, Nov 2025 (revised Jan 2026)**

In practice, modern foundation models are optimized in two tightly-coupled layers:

- **(A) Model-side optimization:** quantization, pruning/sparsity, distillation, low-rank / factorized parameterization, etc. (reduce compute/memory while preserving task performance).
- **(B) Deployment optimization:** compilers + runtimes (e.g., **TensorRT-LLM/TensorRT**, **OpenVINO**, **ONNX Runtime**, **LiteRT** (ex-TFLite), **TVM**, **ncnn**, vendor SDKs) that turn an optimized checkpoint into a hardware-efficient engine.

> NOTE: TorchAO / **torchao** mostly belongs to (A), but it increasingly acts as the “bridge” into (B) via export/compile flows.

This note is a quick mental map of the mainstream pathways for compressing and deploying foundation models (with links to docs + code).

---

## 1. Mainstream **(A) Model-side optimization**

### 1.1 Quantization (almost always the first step)

**Goal:** reduce weights/activations from FP16/FP32 -> INT8/INT4/FP8/FP4 **without** unacceptable accuracy loss.

Common variants (esp. for transformers) (survey: [Zhu et al., 2024][survey-llm-compression]):

- **Post-Training Quantization (PTQ)**  
  No retraining; use a small calibration set.
  - *Static* (typical INT8): collect activation stats offline, quantize weights + activations.
  - *Dynamic* (common on CPUs): quantize weights offline; compute some activation quant params at runtime.
  - Transformer-specific PTQ recipes (examples): **SmoothQuant**, **AWQ**.

  Codebases:
  - SmoothQuant: [mit-han-lab/smoothquant][smoothquant]
  - AWQ: [mit-han-lab/llm-awq][awq]

- **Quantization-Aware Training (QAT)**  
  Simulate quantization during fine-tuning to recover accuracy when PTQ is too lossy.
  - PyTorch-native QAT for low-bit: [torchao QAT docs][torchao-qat]
  - NVIDIA QAT workflows (for TensorRT/TensorRT-LLM): [Model Optimizer docs][modelopt-docs]

- **Mixed precision & vendor-specific formats**
  - Mixed FP16/BF16 is the default “cheap win.”
  - NVIDIA-specific low-bit floats: **FP8** and **NVFP4/FP4** are supported through **TensorRT** + **NVIDIA Model Optimizer (ModelOpt)**.  
    Practical entry points: [NVFP4 overview][nvfp4-blog], [TensorRT quantized types][tensorrt-quantized-types].

> Reality check: the *format* only matters if your deployment stack has the matching kernels (this is where TensorRT-LLM / TensorRT / OpenVINO / ORT / LiteRT differ).

**Example (GR00T-style VLA):**
- Vision + language backbones: INT8 / FP8 / FP4 (where supported)
- Action/diffusion head + final layers: FP16/BF16

References:
- GR00T N1.5 technical page: [research.nvidia.com/labs/gear/gr00t-n1_5][groot-n15]
- Isaac GR00T code/checkpoints: [NVIDIA/Isaac-GR00T][isaac-groot]

---

### 1.2 Pruning & sparsity

**Goal:** remove “unimportant” parameters (often combined with quantization). Speedups depend heavily on **kernel support** and **sparsity structure**.

Common flavors:

- **Unstructured pruning (weight sparsity)**
  - Easy to apply, but usually needs specialized sparse kernels to see wall-clock speedups.

- **Structured pruning (more reliable speedups)**
  - Prune attention heads, MLP channels, entire blocks/layers, tokens, etc.

- **N:M (semi-structured) sparsity**
  - Example: **2:4 sparsity** (50% zeros in a constrained pattern) which maps to NVIDIA Sparse Tensor Cores.
  - Acceleration stack often involves **TensorRT** and/or **cuSPARSELt** plus an export path that preserves sparsity metadata.

  References:
  - cuSPARSELt docs: [docs.nvidia.com/cuda/cusparselt][cusparselt]
  - torchao sparsity overview (incl. semi-structured): [torchao sparsity docs][torchao-sparsity]
  - NVIDIA Model Optimizer (pruning + sparsity): [Model Optimizer repo][modelopt-repo]

---

### 1.3 Knowledge Distillation (KD)

**Goal:** train a smaller/cheaper **student** to mimic a larger **teacher**.

Main flavors:
- **Logit distillation:** student matches teacher soft logits.
- **Feature distillation:** align hidden states / attention maps.
- **Sequence / behavior distillation:** student imitates teacher-generated trajectories/actions.

Tutorials / tooling:
- PyTorch KD tutorial: [Knowledge distillation in PyTorch][pytorch-kd]
- NVIDIA distillation workflow (ModelOpt + NeMo/HF): [Model Optimizer distillation docs][modelopt-distill]

---

### 1.4 Low-rank & factorization tricks

Often used for *parameter-efficient adaptation*, and sometimes for compression if merged or baked into the final model:

- **LoRA / low-rank adapters:** train low-rank deltas, optionally merge.
  - LoRA ref: [microsoft/LoRA][lora]
  - Practical LoRA/QLoRA tooling: [huggingface/peft][peft]

- **Matrix/tensor decompositions:** SVD / Tucker / CP, etc.

---

## 2. Mainstream **(B) Deployment pipelines / toolchains**

### 2.1 NVIDIA-centric: TensorRT-LLM + NVIDIA Model Optimizer (ModelOpt)

**NVIDIA Model Optimizer** (formerly “TensorRT Model Optimizer”) is the main toolkit for PTQ/QAT + pruning + distillation + speculative decoding + sparsity in the NVIDIA stack.

- Code + docs:
  - Repo: [NVIDIA/Model-Optimizer][modelopt-repo]
  - Docs: [nvidia.github.io/Model-Optimizer][modelopt-docs]
- LLM runtime:
  - TensorRT-LLM repo: [NVIDIA/TensorRT-LLM][tensorrt-llm-repo]
  - TensorRT-LLM docs: [nvidia.github.io/TensorRT-LLM][tensorrt-llm-docs]

**Pipeline sketch**
1. Start from a Hugging Face / PyTorch checkpoint (e.g., GR00T N1.5).
2. Apply PTQ or QAT with ModelOpt (INT8/FP8/NVFP4, etc.).
3. If needed: pruning/sparsity + distillation to recover accuracy at lower cost.
4. Export/build a TensorRT(-LLM) engine; deploy (Jetson / server GPUs / Blackwell-class hardware).

Pre-quantized checkpoints:
- Hugging Face collection: [Inference-optimized checkpoints (Model Optimizer)][modelopt-hf-collection]

---

### 2.2 Intel / CPU-centric: OpenVINO + NNCF (plus Intel Neural Compressor where useful)

For Intel CPUs/GPUs and many industrial deployments, the “mainline” path today is:

- **OpenVINO** as the inference runtime: [OpenVINO toolkit][openvino]
- **NNCF** as the compression backend (PTQ/QAT/weight compression):  
  - Docs: [OpenVINO model optimization (NNCF)][openvino-modelopt]
  - Repo: [openvinotoolkit/nncf][nncf-repo]

Hugging Face-friendly workflow:
- Optimum Intel + OpenVINO/NNCF: [Optimum Intel OpenVINO optimization][optimum-intel-openvino]

**Intel Neural Compressor (INC)** is still relevant as a cross-framework quant/prune/distill toolkit (especially outside OpenVINO-only workflows):
- Docs: [intel.github.io/neural-compressor][inc-docs]
- Repo: [intel/neural-compressor][inc-repo]

---

### 2.3 PyTorch-native: torchao + torch.export (PT2E quantization)

If you want to stay close to PyTorch while exploring low-bit + sparsity:

- torchao docs: [pytorch.org/ao][torchao-docs]
- Repo: [pytorch/ao][torchao-repo]

Key tutorials (PyTorch 2 export quantization):
- PTQ (graph-mode): [PyTorch 2 Export PTQ][pt2e-ptq]
- QAT (graph-mode): [PyTorch 2 Export QAT][pt2e-qat]

Pruning in “vanilla PyTorch”:
- `torch.nn.utils.prune` is useful for simple experiments, but for structured pruning (channels/blocks with dependency handling), libraries like [VainF/Torch-Pruning][torch-pruning] are often more practical.

---

### 2.4 Framework-agnostic / edge-oriented runtimes

Common choices across heterogeneous edge targets:

- **ONNX Runtime** (quantization + graph optimizations):  
  - ORT quantization docs: [onnxruntime.ai quantization guide][ort-quant]
- **LiteRT** (formerly TensorFlow Lite) for mobile/embedded:  
  - Overview: [ai.google.dev/edge/litert][litert]
  - GitHub: [google-ai-edge/LiteRT][litert-repo]
- **Apache TVM** (compiler + autotuning): [tvm.apache.org][tvm]
- **ncnn** (lightweight C++ runtime): [Tencent/ncnn][ncnn]

Typical flow:
1. Do pruning/KD in PyTorch/TF.
2. Export (ONNX / LiteRT / IR).
3. Run runtime-specific quantization + graph optimizations.
4. Deploy.

---

### 2.5 “All-in-one” compression frameworks

If you want config-driven automation across methods:

- **DeepSpeed Compression** (quantization + pruning + distillation workflows): [DeepSpeed model compression tutorial][deepspeed-compression]
- **LLM Compressor** (vLLM-focused compression toolkit): [vllm-project/llm-compressor][llm-compressor]
- **Pruna** (commercial + OSS tooling): [Pruna docs][pruna]

Note: older “SparseML” references exist, but the upstream repo is archived; treat it as legacy unless your org already depends on it.

---

### 2.6 Example: GR00T-like robotics FM

A practical “first pass” for a GR00T-style VLA:

1. **Baseline profiling** on target (Jetson / server GPU / CPU box / SoC).
2. **Quantize** most transformer layers (PTQ INT8/FP8/FP4 depending on hardware + stack); keep sensitive heads higher precision.
3. **Structured pruning / 2:4 sparsity** only if your deployment engine has real sparse kernels for your shapes.
4. **Distill**:
   - smaller VLA, or
   - task-specific students (e.g., manipulator-only policy).
5. Export to the deployment stack:
   - NVIDIA path: PyTorch -> ModelOpt -> TensorRT-LLM/TensorRT
   - Intel path: PyTorch/HF -> OpenVINO IR -> NNCF -> OpenVINO runtime
   - General path: PyTorch -> ONNX -> ORT / TVM / ncnn
   - Mobile path: TF/PyTorch -> LiteRT

---

## 3. Serving-time optimization (often the biggest real-world win)

A useful mental model: cost splits into **prefill** (prompt processing) and **decode** (token-by-token).
- **Prefill** is usually **compute-bound** (big GEMMs + attention).
- **Decode** is often **memory / KV-cache bandwidth bound**.

So:
- Weight-only INT4/FP4 helps decode *if* your stack has good kernels.
- Better attention kernels (FlashAttention/FlashInfer) help prefill and reduce memory traffic.
- KV-cache tricks matter most for long context and high concurrency.

### 3.1 Batching + scheduling + KV memory management

If you do nothing else, choose a serving engine that gives you:
- **continuous / dynamic batching**
- **paged KV cache** (reduces fragmentation under concurrency)
- optional **chunked prefill** (smooths very long prompts)

Good entry points:
- **vLLM**: PagedAttention + continuous batching + CUDA/HIP graph execution.  
  Docs: [vLLM docs][vllm-docs] · Repo: [vllm-project/vllm][vllm-repo]
- **Hugging Face Text Generation Inference (TGI)**: production server with dynamic batching + tensor parallelism.  
  Docs: [TGI docs][tgi-docs] · Repo: [huggingface/text-generation-inference][tgi-repo]
- **SGLang**: high-performance LLM serving framework + runtime stack.  
  Repo: [sgl-project/sglang][sglang-repo]

> Reality check: feature parity differs (quant formats, speculative decoding, MoE, multi-modal, etc.).  
> Always confirm against each runtime’s “supported hardware + quantization” tables.

### 3.2 Kernel libraries that matter in practice

For transformer-heavy workloads, **attention + MLP kernels** are usually the make-or-break.

- **FlashAttention** (training + inference attention kernels): [Dao-AILab/flash-attention][flashattn]
- **FlashInfer** (serving-focused kernels: attention, paged attention, sampling, etc.):  
  Repo: [flashinfer-ai/flashinfer][flashinfer] · Docs: [flashinfer.ai docs][flashinfer-docs]

### 3.3 KV-cache optimization for long context + high concurrency

When context length or concurrency grows, KV cache can dominate VRAM and drive latency cliffs.

Two complementary strategies:
- **Systems**: paged KV cache + chunked prefill (runtime feature).
- **Model-side**: KV cache **quantization/compression** (typically 4–8 bit; often mixed precision).

Researchy-but-usable codebases:
- **KVQuant**: [SqueezeAILab/KVQuant][kvquant]
- **ZipCache**: [ThisisBillhe/ZipCache][zipcache]

### 3.4 Decoding acceleration (reduce target-model forward passes)

If decode is the bottleneck, you can reduce the number of expensive target-model steps:
- **Speculative decoding** (draft model + verification): [romsto/Speculative-Decoding][specdec]
- **Multi-token heads** (Medusa): [FasterDecoding/Medusa][medusa]

---

## 4. Deployment lanes by hardware (quick cheat sheet)

### NVIDIA GPUs / Jetson / Blackwell-class

- **NVIDIA Model Optimizer (ModelOpt)** for PTQ/QAT + sparsity/distillation:  
  Docs: [ModelOpt docs][modelopt-docs] · Repo: [NVIDIA/Model-Optimizer][modelopt-repo]
- **TensorRT-LLM** for engine build + kernels + serving:  
  Docs: [TensorRT-LLM docs][tensorrt-llm-docs] · Repo: [NVIDIA/TensorRT-LLM][tensorrt-llm-repo]

### AMD GPUs (ROCm/HIP) + non-NVIDIA datacenter

- Serving engines like **vLLM** can run with **HIP** backends; quantization support is more kernel-dependent and can be narrower than NVIDIA.
- PyTorch **torch.compile** (Inductor) is a good “graph+kernel” optimization baseline across NVIDIA/AMD/Intel GPUs (via Triton):  
  API: [torch.compile][torch-compile] · Guide: [torch.compiler docs][torch-compiler-guide]

### CPUs (x86 + ARM servers)

First levers:
- smaller model (distill) and/or weight-only quantization (INT8/INT4).

Runtimes:
- **OpenVINO + NNCF** (Intel-heavy deployments): [OpenVINO][openvino] · [NNCF repo][nncf-repo]
- **ONNX Runtime** (cross-platform): [ORT quantization docs][ort-quant]
- **ONNX Runtime GenAI** (generation loop tooling): [microsoft/onnxruntime-genai][ort-genai]
- **llama.cpp** (local C++ inference; GGUF ecosystem): [ggml-org/llama.cpp][llamacpp]

### Apple silicon (laptop / mobile-class SoC)

- **MLX** for training + inference on Apple silicon: [MLX repo][mlx] · [MLX docs][mlx-docs]
- LLM-oriented tooling: [mlx-lm][mlx-lm]
- App conversion pipeline: **coremltools**: [Repo][coremltools] · [Guide][coremltools-guide]

### Android / Qualcomm / embedded SoCs

- **LiteRT** (ex-TFLite) runtime + delegates:  
  Docs: [LiteRT][litert] · Repo: [google-ai-edge/LiteRT][litert-repo] · Samples: [litert-samples][litert-samples]  
  LLM pipeline: [google-ai-edge/LiteRT-LM][litert-lm]
- **ExecuTorch** (PyTorch -> on-device runtime): [Docs][executorch] · [Repo][executorch-repo]
- **ONNX Runtime + QNN EP** (Qualcomm acceleration): [ORT QNN EP docs][ort-qnn] · [Qualcomm ORT QNN EP docs][qnn-ort-docs]

### “Runs everywhere” local inference engines

These are often the fastest way to get something working across laptops + edge boxes:
- **MLC LLM** (TVM-based compiler + runtime for LLM deployment): [mlc-ai/mlc-llm][mlc-llm] · [MLC LLM docs][mlc-llm-docs]
- **llama.cpp** (GGUF + broad backend support): [llamacpp][llamacpp]

---

## 5. Beyond LLMs: VLMs + diffusion model optimization in robotics

### VLMs / VLAs

- Optimize **each submodule separately** (vision encoder, LLM, action head) and re-profile end-to-end.
- Watch for non-model bottlenecks: image decode, resizing, tokenization, simulator/robot loop.

### Diffusion / image generation

Two big levers:
- **Reduce steps** (often bigger win than faster steps): Latent Consistency Models (LCM): [luosiallen/latent-consistency-model][lcm]
- **Make each step faster** (quantize/compile kernels):
  - Diffusers bitsandbytes quantization guide: [Diffusers bitsandbytes quantization][diffusers-bnb]
  - Reference implementations / research code: [Stability-AI/generative-models][stability-generative-models]

---

## 6. Minimal “what should I do first?” decision tree

1. **Profile** and label the bottleneck: **weights** vs **KV cache** vs **kernels** vs **scheduling**.
2. If **decode/VRAM** dominates -> start with **weight-only INT4/FP4**, but only if your runtime supports it well.
3. If **long context/concurrency** dominates -> fix **paged KV + chunked prefill**, then consider **KV cache quantization**.
4. If **prefill compute** dominates -> better kernels (FlashAttention/FlashInfer) + compile (torch.compile / TensorRT).
5. If you still can’t hit constraints -> **distill** (often the only way to cut both compute *and* memory).

---

## 7. Closing remarks

- **Model optimization and deployment optimization are inseparable.**
- Most “wins” come from **matching a compression method to the runtime’s kernels**.
- Treat it as a feedback loop: profile -> compress -> compile -> measure -> iterate.

<!-- References / links -->

[survey-llm-compression]: https://aclanthology.org/2024.tacl-1.85.pdf

<!-- Quantization methods -->
[smoothquant]: https://github.com/mit-han-lab/smoothquant
[awq]: https://github.com/mit-han-lab/llm-awq

<!-- NVIDIA stacks  -->
[modelopt-repo]: https://github.com/NVIDIA/Model-Optimizer
[modelopt-docs]: https://nvidia.github.io/Model-Optimizer/
[tensorrt-llm-repo]: https://github.com/NVIDIA/TensorRT-LLM
[tensorrt-llm-docs]: https://nvidia.github.io/TensorRT-LLM/
[tensorrt-quantized-types]: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html
[nvfp4-blog]: https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
[modelopt-hf-collection]: https://huggingface.co/collections/nvidia/inference-optimized-checkpoints-with-model-optimizer

<!-- Sparsity -->
[cusparselt]: https://docs.nvidia.com/cuda/cusparselt/
[torchao-sparsity]: https://docs.pytorch.org/ao/stable/sparsity.html

<!-- Distillation -->
[pytorch-kd]: https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html
[modelopt-distill]: https://nvidia.github.io/Model-Optimizer/guides/distillation.html

<!-- LoRA / PEFT -->
[lora]: https://github.com/microsoft/LoRA
[peft]: https://github.com/huggingface/peft

<!-- Intel stacks -->
[openvino]: https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html
[openvino-modelopt]: https://docs.openvino.ai/2025/openvino-workflow/model-optimization.html
[nncf-repo]: https://github.com/openvinotoolkit/nncf
[optimum-intel-openvino]: https://huggingface.co/docs/optimum-intel/en/openvino/optimization
[inc-docs]: https://intel.github.io/neural-compressor/
[inc-repo]: https://github.com/intel/neural-compressor

<!-- PyTorch-native optimization -->
[torchao-docs]: https://pytorch.org/ao/
[torchao-repo]: https://github.com/pytorch/ao
[torchao-qat]: https://docs.pytorch.org/ao/stable/api_ref_qat.html
[pt2e-ptq]: https://docs.pytorch.org/ao/stable/tutorials_source/pt2e_quant_ptq.html
[pt2e-qat]: https://docs.pytorch.org/ao/stable/tutorials_source/pt2e_quant_qat.html
[torch-pruning]: https://github.com/VainF/Torch-Pruning

<!-- General runtimes -->
[ort-quant]: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
[litert]: https://ai.google.dev/edge/litert
[litert-repo]: https://github.com/google-ai-edge/LiteRT
[tvm]: https://tvm.apache.org/
[ncnn]: https://github.com/Tencent/ncnn

<!-- Automation frameworks -->
[deepspeed-compression]: https://www.deepspeed.ai/tutorials/model-compression/
[llm-compressor]: https://github.com/vllm-project/llm-compressor
[pruna]: https://docs.pruna.ai/en/stable/compression.html

<!-- GR00T -->
[groot-n15]: https://research.nvidia.com/labs/gear/gr00t-n1_5/
[isaac-groot]: https://github.com/NVIDIA/Isaac-GR00T

<!-- Serving engines / kernels -->
[vllm-docs]: https://docs.vllm.ai/en/latest/
[vllm-repo]: https://github.com/vllm-project/vllm
[tgi-docs]: https://huggingface.co/docs/text-generation-inference/en/index
[tgi-repo]: https://github.com/huggingface/text-generation-inference
[sglang-repo]: https://github.com/sgl-project/sglang
[flashattn]: https://github.com/Dao-AILab/flash-attention
[flashinfer]: https://github.com/flashinfer-ai/flashinfer
[flashinfer-docs]: https://docs.flashinfer.ai/

<!-- KV cache + decoding acceleration -->
[kvquant]: https://github.com/SqueezeAILab/KVQuant
[zipcache]: https://github.com/ThisisBillhe/ZipCache
[specdec]: https://github.com/romsto/Speculative-Decoding
[medusa]: https://github.com/FasterDecoding/Medusa

<!-- PyTorch compile / export -->
[torch-compile]: https://docs.pytorch.org/docs/stable/generated/torch.compile.html
[torch-compiler-guide]: https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler.html

<!-- Local inference engines -->
[llamacpp]: https://github.com/ggml-org/llama.cpp
[ort-genai]: https://github.com/microsoft/onnxruntime-genai
[mlc-llm]: https://github.com/mlc-ai/mlc-llm
[mlc-llm-docs]: https://llm.mlc.ai/

<!-- Apple / Core ML -->
[mlx]: https://github.com/ml-explore/mlx
[mlx-docs]: https://ml-explore.github.io/mlx/
[mlx-lm]: https://github.com/ml-explore/mlx-lm
[coremltools]: https://github.com/apple/coremltools
[coremltools-guide]: https://apple.github.io/coremltools/docs-guides/

<!-- LiteRT / ExecuTorch / Qualcomm -->
[litert-samples]: https://github.com/google-ai-edge/litert-samples
[litert-lm]: https://github.com/google-ai-edge/LiteRT-LM
[executorch]: https://docs.pytorch.org/executorch/index.html
[executorch-repo]: https://github.com/pytorch/executorch
[ort-qnn]: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
[qnn-ort-docs]: https://docs.qualcomm.com/bundle/publicresource/topics/80-62010-1/ort-qnn-ep.html

<!-- Diffusion / VLM extras -->
[lcm]: https://github.com/luosiallen/latent-consistency-model
[diffusers-bnb]: https://huggingface.co/docs/diffusers/en/quantization/bitsandbytes
[stability-generative-models]: https://github.com/Stability-AI/generative-models
