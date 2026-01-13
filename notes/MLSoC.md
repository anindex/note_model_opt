# ML Training & Inference on SoCs: the Other-than-CUDA Landscape

**An T. Le, Hanoi, Dec 2025 (revised Jan 2026)**

On NVIDIA, the world is unusually simple: most performance work bottoms out in **CUDA**.

On **ARM-based SoCs** (phones, tablets, SBCs, embedded boards), there is no single “CUDA for everything”. What you get instead is a *layered stack*:

1. **Low-level compute APIs** (GPU/NPU “assembly language”)
2. **CPU/GPU kernel libraries** (cuBLAS/cuDNN analogues)
3. **Graph runtimes + ML compilers** (the “TensorRT/XLA-ish” layer)
4. **Vendor SDKs** that glue accelerators into something usable

This note is a *practical map* of that ecosystem with codebases + docs you can jump into quickly.

---

## 0. A quick “what stack should I touch?” cheat sheet

### If you ship an Android app (phones/tablets)
- Start: **LiteRT** (ex‑TensorFlow Lite) delegates (GPU / vendor NPUs), or **ONNX Runtime Mobile**
- If you’re PyTorch-first: **ExecuTorch**
- Only go low-level (Vulkan/OpenCL) if you’re building a custom runtime or kernels.

### If you ship on Apple silicon (iPhone/iPad/Mac)
- Deployment: **Core ML** (best path to Apple Neural Engine)
- Research / local fine-tuning: **MLX** (CPU+GPU via Metal; great ergonomics)
- Low-level: **Metal** / **MPS** if you’re writing custom kernels.

### If you ship embedded Linux boards (robotics / edge boxes)
- Start with the **vendor SDK** (TI TIDL, Qualcomm QNN, NXP eIQ, Rockchip RKNN, …)
- If you want one runtime across boards: **TVM / IREE / ONNX Runtime**
- For local LLMs on ARM: **llama.cpp**, **MLC‑LLM**, **MNN**.

### If you ship on microcontrollers (TinyML)
- Start: **LiteRT for Microcontrollers**, **CMSIS‑NN**, optionally **Ethos‑U + Vela**
- TVM path: **microTVM** (if you want compiler-driven deployment).

---

## 1. Low-level compute APIs: the “CUDA-ish” layer

These are the primitives everything else eventually targets.

### Vulkan compute (SPIR‑V) + Kompute

For **mobile GPUs** (Adreno, Mali, etc.), modern cross-vendor compute is typically **Vulkan** compute shaders.

If you don’t want raw Vulkan boilerplate:
- **Kompute (Vulkan compute framework):** https://github.com/KomputeProject/kompute  
  A nice “CUDA-ish” abstraction layer with cross-platform dispatch helpers.

When Vulkan matters:
- Custom ops (vision/physics/signal processing)
- Non-standard kernels that delegates/runtimes don’t support
- You need one GPU backend across many Android devices

### OpenCL (still relevant nowadays, but driver-dependent)

OpenCL is “older” than Vulkan compute, but it’s still relevant:
- **ARM Compute Library (ACL)** uses OpenCL for Mali GPU acceleration paths.
- Some edge inference stacks and research code still target OpenCL directly.

If you can choose today: prefer Vulkan for “future-proof” Android GPU work, but keep OpenCL in your toolbox when the vendor stack expects it.

### Apple: Metal + MPS

On Apple SoCs, **Metal** is the low-level GPU API and **Metal Performance Shaders (MPS)** is the “fast kernels” layer:
- Metal: https://developer.apple.com/metal/
- MPS: https://developer.apple.com/documentation/metalperformanceshaders

---

## 2. Kernel libraries: cuBLAS/cuDNN analogues (CPU-first on SoCs)

### ARM Compute Library (ACL)

- **Repo:** https://github.com/ARM-software/ComputeLibrary  
  Hand-tuned primitives for ARM CPUs (NEON/SVE/SVE2) and Mali GPUs (OpenCL).

ACL is often *indirectly* consumed via higher runtimes (ARM NN, frameworks, vendor SDKs).

Good starting points:
- ACL docs + build guides live in the repo (see “Documentation” section in README).
- ARM tutorial referenced by the repo: “AlexNet on Raspberry Pi” (linked from ACL README).

### ARM NN (Arm’s inference runtime)

- **Repo:** https://github.com/ARM-software/armnn  
- **Docs:** https://arm-software.github.io/armnn/

> Key reality check (2026):
> - ARM NN’s **recommended integration** is via the **TF Lite Delegate**.
> - It supports **TensorFlow Lite** and **ONNX** models (parsers exist but have less coverage than the delegate).
> - Ethos integration is primarily **Ethos‑N** (Linux-class NPUs). For **Ethos‑U** (microcontrollers), you usually go through LiteRT Micro + Vela, not ARM NN.

If you’re on Android and want “Arm acceleration without writing kernels”:
- ARM also maintains an Android NN driver/HAL integration for ARM IP (see ARM NN README links).

### XNNPACK (mobile CPU inference workhorse)

- **Repo:** https://github.com/google/XNNPACK  
XNNPACK is a highly-optimized CPU operator library widely used under-the-hood (not usually called directly). It shows up as:
- LiteRT/TFLite CPU backend
- ONNX Runtime’s **XNNPACK Execution Provider**: https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html

### oneDNN on AArch64

- **Repo:** https://github.com/uxlfoundation/oneDNN  
oneDNN supports **AArch64** and is a common “fast CPU math” backend in larger stacks.

### ARM KleidiAI (new-ish CPU micro-kernel layer)

- **Repo:** https://github.com/ARM-software/kleidiai  
KleidiAI provides optimized micro-kernels (notably relevant for low-bit GEMM/GEMV patterns showing up in LLM inference).

### TinyML kernels: CMSIS‑NN

For Cortex‑M microcontrollers:
- **CMSIS‑NN:** https://github.com/ARM-software/CMSIS-NN  
This is the “fast int8 kernels” layer for TinyML deployments.

---

## 3. Graph runtimes & ML compilers: the “TensorRT/XLA” layer for SoCs

These sit above low-level APIs and try to give you portability + performance.

### LiteRT (formerly TensorFlow Lite)

- **Overview:** https://ai.google.dev/edge/litert/overview  
- **Repo:** https://github.com/google-ai-edge/LiteRT  
- **GPU delegate docs:** https://ai.google.dev/edge/litert/performance/gpu  
- **Microcontrollers:** https://ai.google.dev/edge/litert/microcontrollers/overview

Use LiteRT when:
- You want the “default Android/embedded path”
- You rely on delegates (GPU, vendor NPU integrations, etc.)
- You want strong tooling around conversion and mobile deployment

### ExecuTorch (PyTorch on-device runtime)

- **Main docs:** https://docs.pytorch.org/executorch/
- **Qualcomm QNN backend tutorial:** https://docs.pytorch.org/executorch/stable/backends-qualcomm.html
- **MediaTek backend:** https://docs.pytorch.org/executorch/1.0/backends-mediatek.html

ExecuTorch is a good “PyTorch-native” way to export + run models on phones/edge devices, while still hitting vendor accelerators.

### ONNX Runtime (ORT) on mobile + edge

- **ORT Mobile:** https://onnxruntime.ai/docs/get-started/with-mobile.html  
- **Execution providers overview:** https://onnxruntime.ai/docs/execution-providers/  
- **QNN EP:** https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html  
- **XNNPACK EP:** https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html  

LLM-specific:
- **onnxruntime-genai (generate loop + KV cache management):** https://github.com/microsoft/onnxruntime-genai  
- Docs: https://onnxruntime.ai/docs/genai/

ORT is especially useful when:
- Your model pipeline is ONNX-first
- You want to swap acceleration backends via EPs (QNN, CoreML, XNNPACK, etc.)
- You want a single engine across Android + embedded Linux

### Apache TVM (+ microTVM)

- **TVM:** https://tvm.apache.org/  
- **microTVM design doc:** https://tvm.apache.org/docs/arch/microtvm_design.html  

TVM is compelling when:
- You want a programmable compiler (schedules, auto-tuning, custom backends)
- You’re spanning heterogeneous devices (CPU + GPU + NPU)

### IREE (MLIR-based compiler + runtime)

- **IREE:** https://iree.dev/  
IREE lowers via MLIR to backends including CPU and Vulkan (mobile GPUs), aiming for a unified compiler toolchain.

### “LLM-first on device”: MLC‑LLM, llama.cpp, ncnn, MNN

If your *main* workload is **LLM inference on edge/mobile**:

- **vLLM (CPU on ARM / server-class ARM):** https://docs.vllm.ai/en/latest/getting_started/installation/cpu/ 
  A high-throughput serving engine; on ARM it targets the **CPU backend** (NEON), useful for *server-class* AArch64 or beefy edge boxes.

- **MLC‑LLM:** https://llm.mlc.ai/  
  Uses “TVM Unity” to compile models; targets Metal/Vulkan/OpenCL backends depending on platform.

- **llama.cpp:** https://github.com/ggml-org/llama.cpp  
  Practical, minimal-dependency LLM inference across CPU (NEON/SVE), Metal, Vulkan/OpenCL (varies by build/target).

- **ncnn:** https://github.com/Tencent/ncnn  
  Mobile-first C++ inference engine with a strong Vulkan path (popular for CV models on Android).

- **MNN:** https://github.com/alibaba/MNN  
  Lightweight engine with strong on-device focus (and a lot of real-world Android usage inside Alibaba’s ecosystem).

---

## 4. Vendor SoC stacks: “CUDA for X”

This layer is where you get the best perf/W — but also the most vendor specificity.

### Qualcomm (Snapdragon / QRB / RB-class robotics)

- **Qualcomm AI Engine Direct / QNN SDK:** https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk  
Integration points:
- ExecuTorch QNN backend (docs above)
- ORT QNN EP (docs above)

### MediaTek (NeuroPilot)

- **NeuroPilot portal:** https://neuropilot.mediatek.com/  
- LiteRT has an explicit guide for NeuroPilot integration: https://ai.google.dev/edge/litert/next/mediatek

### TI (Jacinto / AM68A / AM69A)

- **TI Edge AI SDK:** https://software-dl.ti.com/jacinto7/esd/processor-sdk-linux-am68a/latest/exports/docs/linux/index_Edge_AI.html  
- **TIDL tools:** https://github.com/TexasInstruments/edgeai-tidl-tools  

TI’s stack is strong when you need a full capture→preprocess→infer pipeline (often with GStreamer/OpenVX integration).

### NXP i.MX (eIQ)

- **eIQ environment:** https://www.nxp.com/design/design-center/software/eiq-ai-development-environment%3AEIQ  
- **i.MX Machine Learning User’s Guide (UG10166):** https://www.nxp.com/docs/en/user-guide/UG10166.pdf  

> Reality check 2026: vendors sometimes remove/shift supported frontends over time. UG10166 notes removed components (e.g., TensorFlow parser) and current supported parsers/runtime paths.

### Rockchip (RK3588, etc.)

- **RKNN Toolkit2:** https://github.com/rockchip-linux/rknn-toolkit2  
- **RKNPU2 runtime:** https://github.com/rockchip-linux/rknpu2  
- **RKNN‑LLM:** https://github.com/airockchip/rknn-llm  

This matters because Rockchip boards are common in robotics/SBC deployments, and RKNN is often the practical way to hit the onboard NPU.

---

## 5. Cross-vendor Android abstraction: NNAPI (deprecated, still encountered)

- **NNAPI docs:** https://developer.android.com/ndk/guides/neuralnetworks  
- **Migration guide:** https://developer.android.com/ndk/guides/neuralnetworks/migration-guide  

As of Android 15, **the NNAPI NDK API is deprecated**. In practice:
- Existing devices and drivers still exist, so you may still *encounter* NNAPI.
- For forward-looking work, prefer **LiteRT delegates** or vendor SDKs directly (QNN, NeuroPilot, etc.).

---

## 6. Training vs inference on SoCs: what’s realistic in early 2026?

### Inference: the default SoC use-case
SoCs shine at **efficient inference** (power/thermal constraints, on-device privacy).

### Training: mostly “lightweight” (with a few real exceptions)
What’s realistic:
- Small fine-tunes (adapters/LoRA), small heads, online updates
- On Apple silicon: MLX makes *real* training/fine-tuning practical on-device
- On Android/embedded: “on-device training” exists in LiteRT docs, but it’s niche and often bounded to smaller models / specific workflows

Most mobile NPUs are still inference-centric from an exposed API standpoint.

---

## 7. A practical deployment loop that works across SoCs

1. **Pick your interchange format**
   - Android/embedded SoCs: `.tflite` (LiteRT) or `.onnx` (ORT)
   - Apple: Core ML (`.mlpackage`/`.mlmodel`) via coremltools
   - LLM edge: GGUF/GGML (llama.cpp) or MLC formats (MLC‑LLM)

2. **Use the highest-level runtime that still hits your accelerator**
   - LiteRT delegate / ExecuTorch backend / ORT EP
   - Vendor SDK when you need max perf-per-watt

3. **Measure delegate/EP coverage**
   - “Fast path” is only fast if most ops land on the accelerator
   - Silent fallbacks to CPU are the most common perf surprise

4. **Only drop to Vulkan/OpenCL when you *must***
   - Custom ops
   - Research runtimes
   - You own the full deployment binary and want full control

---

## Related notes in this repo

- [Optimizing Models: A Train Of Thought](./OptimizingModels.md) (quantization/pruning/distillation + toolchains)
- [Model Optimization in 2025-2026: A Survey](./ModelOptDeepDive.md) (SOTA infra + codebases)

