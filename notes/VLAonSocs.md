# VLA on SoCs: Deployment Notes (Landscape & Practical)

**Ho T. Hung, An T. Le**

**Hanoi, Jan 2026**

*Scope:* **Vision-Language-Action (VLA)** *deployment* (inference-first) for robotics / embodied agents on **edge SoCs** (phones/tablets, laptops-as-edge, SBCs, Jetson-class modules).  
We cover: (1) model shapes (end-to-end VLA vs split VLM+policy), (2) optimization (quantization, pruning, distillation, cache/scheduling tricks), and (3) runtime lanes (Core ML/ANE, QNN, TensorRT, RKNN, LiteRT/ORT/OpenVINO) + profiling/packaging.  

---

## 0) TL;DR

- **Best “quick win” today:** run a **sub‑1B VLA** end-to-end (e.g., **SmolVLA‑450M**) or run a **VLM/LLM + smaller action head**; avoid “full-size” VLAs unless you have a Jetson AGX‑class device.
- **Main bottleneck:** *vision tokens* + *autoregressive decoding* (and/or multi-step diffusion).  
  Your biggest speedups usually come from:
  - **Lower image resolution / fewer frames**,  
  - **Pruning/caching visual tokens**,  
  - **Quantizing weights** (4–8 bit),  
  - **Action chunking + async control** (decouple policy rate from control rate).
- **Deployment reality:** You’ll typically mix runtimes:
  - Vision encoder on **GPU/NPU** (TensorRT / Core ML / QNN / RKNN),
  - LLM/VLM on **GPU** (Metal/CUDA) or **NPU** when supported,
  - Action head often stays on **GPU/CPU** unless it’s standard ops.

---

## Living lists for efficient VLA

- **Awesome-VLA (includes an “Efficient-VLA” section):** https://github.com/KwanWaiPang/Awesome-VLA
- **Awesome Efficient VLA (taxonomy + code links):** https://github.com/guanweifan/awesome-efficient-vla
- **Survey (quick landscape):** *A Survey on Efficient Vision-Language-Action Models* - https://arxiv.org/abs/2510.24795

---

## 1) Reference VLA architectures

### A) Autoregressive VLM -> discrete action tokens
**Examples:** OpenVLA, SmolVLA, many “token-based” VLAs  
- Encode image(s) -> visual tokens  
- Condition on instruction -> decode **action tokens** (often 7–DoF or chunked action sequences)
- Great for *generalization*, but can be **decode-bound** and **vision-token heavy**.

### B) VLM + diffusion / flow-matching action head
- VLM provides task context; diffusion/flow head generates continuous actions.
- Often more sample efficient / smooth (improving with recent ideas such as [Real-time Action Chunking](https://arxiv.org/abs/2506.07339)), but **multi-step inference** can be too slow unless distilled (see Sec. 6).

### C) Split pipeline (often the most SoC-friendly)
- **On-device perception** (small VLM/ViT) + **tiny policy/controller**
- Optional: off-device LLM “planner” (cloud or nearby GPU)  
- Sacrifices some end-to-end purity but is usually easier to ship.

### D) Dual-system VLA (slow reasoning + fast control)

This is the “split pipeline” idea *but formalized*: a slow **System 2** reasons, a fast **System 1** executes (often at higher Hz), sometimes with partial parameter sharing.

**Open implementations / references (usable code):**
- **Fast-in-Slow (FiS-VLA):** https://github.com/CHEN-H01/Fast-in-Slow  (project: https://fast-in-slow.github.io/)
- **OpenHelix:** https://github.com/OpenHelix-Team/OpenHelix  (project: https://openhelix-robot.github.io/)
- **Hume:** https://github.com/hume-vla/hume  (project: https://hume-vla.github.io/)

---

## 2) SoC landscape (pragmatic lanes)

> **Rule of thumb:** pick the lane that matches your target device *first*, then pick a model + optimization strategy that the lane supports.

| Lane (typical devices) | Best-supported acceleration stack | What usually fits | Main gotchas |
|---|---|---|---|
| **Apple Silicon** (M‑series Macs, iPhone A‑series) | **Core ML** (ANE/GPU) via `coremltools`, + **MLX** / **MLC‑LLM** (Metal) | 1–7B LLMs on GPU; smaller VLM/VLA on-device | Core ML op coverage & model structure constraints; ANE best with int8 activations |
| **Qualcomm Snapdragon / QCS / RBx** | **QNN** via Qualcomm AI Hub, ORT **QNN EP**, **ExecuTorch QNN**, **LiteRT‑LM** | Small–mid LLMs/VLMs when op coverage is good; “static-ish” graphs | Compiler + op support constraints; Android packaging; memory bandwidth |
| **NVIDIA Jetson Orin** | **TensorRT** (+ TensorRT‑LLM branch), **NanoLLM**, CUDA kernels | Larger VLM/VLA possible (AGX Orin best) | Power/thermals; container compatibility; TensorRT op gaps |
| **Rockchip RK35xx** (RK3588, RK3576…) | **RKNN/RKLLM** toolchain | Small CV models; some LLMs via RKLLM | Toolchain/version churn; limited op support; quantization constraints |
| **Other edge NPUs** (MediaTek / Samsung / Intel / AMD / NXP…) | Usually **LiteRT / NNAPI / OpenVINO / Vulkan / vendor SDK** | Smaller models; mixed success for LLM/VLM | Ecosystem fragmentation; driver/EP maturity |

If you’re new: start with **Apple** (fast iteration) or **Jetson** (best “it just runs” for robotics). Qualcomm is excellent when you align with the **QNN-supported** subgraph.

---

## 3) Lane-specific toolchains

### Apple (Core ML + Metal)
- **Core ML Tools (quantization / pruning / palettization):**  
  - Quantization overview (supports **8-bit and 4-bit weights**, optional **8-bit activations**): https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html  
  - Quantization algorithms (RTN, GPTQ, QAT): https://apple.github.io/coremltools/docs-guides/source/opt-quantization-algos.html
- **MLX (Metal-first research stack):** https://github.com/ml-explore/mlx  
  - MLX-LM (practical LLM inference): https://github.com/ml-explore/mlx-lm
- **MLC-LLM (Metal runtime for LLMs/VLMs):** https://llm.mlc.ai/docs/deploy/ios.html
- **vLLM on Apple Silicon (community):** https://github.com/vllm-project/vllm-metal

**Caveat:** MLC/MLX primarily use **Metal GPU**, not the ANE; ANE acceleration generally comes through **Core ML**.

---

### Qualcomm (QNN ecosystem)
- **Qualcomm AI Hub (models + compilation):** https://aihub.qualcomm.com/get-started  
  - Example on-device model pages:  
    - Llama 3.2 3B Instruct: https://aihub.qualcomm.com/models/llama_v3_2_3b_instruct  
    - Qwen2.5 7B Instruct: https://aihub.qualcomm.com/models/qwen2_5_7b_instruct
- **ONNX Runtime QNN Execution Provider docs:**  
  - QNN EP overview: https://docs.qualcomm.com/bundle/publicresource/topics/80-62010-1/ort-qnn-ep.html  
  - ORT “build model assets for Snapdragon NPU”: https://onnxruntime.ai/docs/genai/howto/build-models-for-snapdragon.html
- **ExecuTorch + QNN backend:**  
  - Build/run Qualcomm backend: https://docs.pytorch.org/executorch/0.4/build-run-qualcomm-ai-engine-direct-backend.html
  - Llama 3 3B tutorial (Android + QNN): https://docs.pytorch.org/executorch/1.0/llm/build-run-llama3-qualcomm-ai-engine-direct-backend.html
- **LiteRT-LM (Google AI Edge, cross-platform LLM runtime):**  
  - Repo: https://github.com/google-ai-edge/LiteRT-LM  
  - NPU guide: https://ai.google.dev/edge/litert/next/litert_lm_npu

**Caveat:** QNN works best when your model can be lowered to a **supported static subgraph** (ops + shapes). Plan for fallbacks (GPU/CPU) for unsupported pieces.

---

### NVIDIA Jetson Orin (CUDA + TensorRT)
- **Jetson AI Lab 2.0 (curated tutorials):** https://www.jetson-ai-lab.com/  
  - **OpenVLA guide (archived / deprecated but still useful):** https://www.jetson-ai-lab.com/archive/openvla.html
- **OpenVLA project page:** https://openvla.github.io/  
- **Jetson Platform Services (VLM inference service):** https://docs.nvidia.com/jetson/jps/inference-services/vlm.html
- **Isaac ROS DNN inference (TensorRT/Triton nodes):** https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference

**Caveat:** TensorRT-LLM support on Jetson exists but is more “branch / JetPack-coupled” than desktop; expect friction and pin versions.

---

### Rockchip (RKNN / RKLLM)
- **RKNN-LLM stack (Rockchip):** https://github.com/airockchip/rknn-llm  
- **RKNN-Toolkit2 (conversion/quant/inference):** https://github.com/rockchip-linux/rknn-toolkit2  
- **RKNN model zoo:** https://github.com/airockchip/rknn_model_zoo

**Caveat:** Expect frequent version changes and stricter conversion constraints than CUDA/Metal stacks.

---

## 3.1 Common model artifacts

When people say “deploy the VLA”, it often means producing multiple artifacts:

- **PyTorch checkpoint** (`.pt/.pth`) - research + baseline correctness
- **ONNX** (`.onnx`) - common interchange, especially for ORT / QNN EP
- **Core ML** (`.mlpackage` / `.mlmodel`) - iOS/macOS app bundle, ANE/GPU targeting
- **QNN / QAIRT context binaries** - Snapdragon NPU-deployable compiled assets
- **TensorRT engines** (`.plan`) - Jetson GPU-optimized engines (build on-device or in matching containers)
- **RKNN / RKLLM formats** - Rockchip-converted assets for NPU/runtime
- **GGUF / GGML-family** - CPU/GPU inference stacks like llama.cpp (useful for LLM-only subsystems)
- **1-bit quantized (BitNet)** - extreme CPU-only quantization for LLM backbones; requires QAT retraining but delivers massive speedups + energy savings (Microsoft BitNet b1.58 ecosystem)

Practical implication: plan for **multiple conversion + calibration passes**, and budget time for “op coverage” debugging.

---

## 3.2 Profiling & debugging

You’ll want both **latency** and **power/thermals**:

- **Apple:** Xcode Instruments (Time Profiler), Metal System Trace, Core ML model profiling tools
- **Android/Qualcomm:** Perfetto/Systrace, Snapdragon Profiler (where available), QNN profiling logs, ORT profiling
- **Jetson:** `tegrastats`, `jtop`, Nsight Systems/Compute, TensorRT verbose logs
- **General:** measure p50/p95 latency and jitter; avoid “one warm run” benchmarks

---

## 4) “Can I run a VLA on-device?” (reality check)

### Large baseline: OpenVLA (7B-class)
- **OpenVLA** is an open-source 7B VLA pretrained on Open X-Embodiment episodes: https://openvla.github.io/
- On Jetson AGX Orin-class devices, the Jetson AI Lab archive reports **INT4/FP8/FP16** runs with a NanoLLM pipeline and publishes example latency/FPS/accuracy numbers: https://www.jetson-ai-lab.com/archive/openvla.html

**Takeaway:** feasible on **AGX Orin**; usually too heavy for phones/SBCs unless you heavily compromise (aggressive quantization + token reduction + low FPS).

---

## 5) Small-scale VLAs worth looking at (SoC-friendly)

> For a “single-device demo” on consumer hardware, these are more realistic than 7B+ VLAs.

### 5.1 SmolVLA‑450M (Hugging Face / LeRobot) - practical baseline
- **Size:** ~450M params  
- **Backbone:** SmolVLM2 (vision encoder) + SmolLM2 (language decoder), trained as a robotics policy  
- **Action head:** flow-matching transformer (for action prediction)  
- **Why it’s good for SoCs:** small enough that quantization + token reduction often gets you into usable latency.
- **Resources:**  
  - Blog: https://huggingface.co/blog/smolvla  
  - Paper: https://arxiv.org/abs/2506.01844  
  - Codebase: https://github.com/huggingface/lerobot

**Deployment hint:** treat it like 3 deployable chunks: {vision encoder, language core, action head}. You may accelerate them with different runtimes.

---

### 5.2 VLA-Adapter (tiny-scale VLA paradigm)
- Project / paper hub: https://vla-adapter.github.io/
- Code: https://github.com/OpenHelix-Team/VLA-Adapter  
- Why it’s useful for edge: focuses on **tiny-scale** designs and practical training/finetuning recipes.

---

### 5.3 NORA (open VLA baseline with released code)
- Code: https://github.com/declare-lab/nora  
- Why it’s useful for edge: a practical open VLA codebase/checkpoints you can inspect + benchmark.

---

### 5.4 Evo-1 (lightweight VLA)
- Code: https://github.com/MINT-SJTU/Evo-1  
- Why it’s useful for edge: designed for **efficiency** while preserving instruction/vision alignment.

---

### 5.5 EdgeVLA (explicit “edge VLA” experiments)
- Code: https://github.com/kscalelabs/evla  
- Why it’s useful for edge: explores training VLAs on **small language models** (e.g., Qwen2-class) and non‑autoregressive objectives.

---

### 5.6 FLOWER (rectified-flow action expert; efficient multi-step policy)
- Code (CALVIN / LIBERO): https://github.com/intuitive-robots/flower_vla_calvin  
- Code (pretraining on OXE): https://github.com/intuitive-robots/flower_vla_pret  
- Why it’s useful for edge: flow-style action heads can be distilled / step-reduced more naturally than long autoregressive decoding.

---

### 5.7 NinA (normalizing-flow action expert; FLOWER variant)
- Code: https://github.com/dunnolab/NinA  
- Why it’s useful for edge: swaps diffusion-style sampling for **normalizing flows** to reduce inference cost.

---

### 5.8 NanoVLA (routing + decoupling for edge) - *paper-only*
- Paper: https://arxiv.org/abs/2510.25122  
- Useful ideas (even without code): **late fusion** (decouple V/L), **routing**, and **chunking** to reduce edge cost.

---

### 5.9 TinyVLA (tiny-scale policy learning)
- Paper: https://arxiv.org/abs/2409.12514  
- Code: https://github.com/JayceWen/tinyvla

---

### 5.10 RoboMamba (state-space / Mamba-style efficiency)
- Paper: https://arxiv.org/abs/2406.04339  
- Project: https://robomamba.github.io/

---

### 5.11 MoLe‑VLA (dynamic layer skipping)
- Paper: https://arxiv.org/abs/2503.20384  
- Code: https://github.com/RoyZry98/MoLe-VLA-Pytorch

### 5.12 SimVLA (minimal design)
- Paper: https://arxiv.org/pdf/2602.18224
- Code: https://github.com/LUOyk1999/SimVLA

## 6) Optimization playbook (SoC-focused)

### Step 0 - Measure end-to-end (not just tokens/sec)
Track **control-loop metrics**:
- Policy latency (p50/p95), jitter
- Action update rate achieved on robot
- Closed-loop success rate (simulation + real)
- Power/thermals (sustained)

### Step 1 - Make the model “edge-shaped”
Low-risk levers:
- Reduce **image resolution** (e.g., 224² -> 160²) *if success rate holds*
- Reduce **frame rate** and use **action chunking**
- Prefer **single image** or a small temporal window (avoid long video token streams)
- Enforce **static shapes** where your compiler needs it (QNN / TensorRT / Core ML)

### Step 2 - Quantize (usually the biggest memory win)
- LLM/VLM: int8 / int4 weights where kernels exist (platform-dependent)  
- Vision encoder: often int8-friendly; keep normalization consistent  
- Action head: fp16/bf16 often fine; quantize only if it’s a bottleneck

Apple note: Core ML tools explicitly supports **4-bit and 8-bit weight quantization** (and optional 8-bit activations).  
Qualcomm note: QNN compilation/EP often expects **quantized** graphs to reach NPU speedups.

VLA-specific quantization (recommended first step):
- **Scale-Calibrated Post-Training Quantization for VLA:** QuantVLA - https://arxiv.org/abs/2602.20309

CPU-only edge note: for **CPU-only edge deployment** (no GPU/NPU), consider **1-bit quantization (BitNet b1.58)** if you can retrain: massive speedups (2.37x–6.17x on x86, 1.37x–5.07x on ARM) + energy savings (72–82% on x86, 55–70% on ARM). See [microsoft/BitNet][bitnet-repo] and OptimizingModels note.

- **Combined quant + token pruning (training-free):** SQAP‑VLA - https://arxiv.org/abs/2509.09090 (code: https://github.com/ecdine/SQAP-VLA)

### Step 3 - Cut visual tokens (often the biggest latency win)
Common patterns:
- Token pruning (static or per-layer)
- Token caching across frames (robotics has high temporal redundancy)
- Late fusion / decoupling so you can reuse parts

Actionable starting points (with code):
- **Token caching across frames:** VLA‑Cache - https://arxiv.org/abs/2502.02175 (code: https://github.com/siyuhsu/vla-cache)
- **Quantization-aware pruning + token pruning:** SQAP‑VLA - https://arxiv.org/abs/2509.09090 (code: https://github.com/ecdine/SQAP-VLA)

Efficient token importance estimation (kernel library):
- **Flash-ColReduce:** Triton kernels for column-wise attention reductions (sum/mean/max) with O(N) memory instead of O(N²); identifies which tokens matter most without materializing full attention; used in visual token pruning (e.g., SparseVILA).
  Code: https://github.com/z-lab/flash-colreduce

If you want a longer (fast-changing) list, start from:
- https://github.com/KwanWaiPang/Awesome-VLA (see “Efficient‑VLA”)
- https://github.com/guanweifan/awesome-efficient-vla

### Step 4 - Make action tokens efficient (quality + speed)
If action discretization is hurting dexterity or sequence length:
- **FAST / FAST+ action tokenization:** https://arxiv.org/abs/2501.09747  
- **Vector-quantized action tokenizers:** VQ‑VLA - https://arxiv.org/abs/2507.01016 (code: https://github.com/xiaoxiao0406/VQ-VLA)

### Step 5 - Reduce decoding overhead
- **Parallel decoding for action chunking:** PD‑VLA - https://arxiv.org/abs/2503.02310 (*paper-only; use as an idea bucket*)
- **Early-exit decoding + consistency distillation:** CEED‑VLA - https://arxiv.org/abs/2506.13725
  Code: https://github.com/OpenHelix-Team/CEED-VLA
  Project: https://irpn-eai.github.io/CEED-VLA/

VLM-heavy VLA note (if you have a large language decoder):
- **Block diffusion parallel drafting :**  DFlash
  Code: https://github.com/z-lab/dflash

### Step 6 - If you have diffusion policies, distill them
Multi-step diffusion is often the blocker for real-time on SoCs. Two complementary approaches:

**Model-side (distillation):**
- **One-step diffusion distillation:** OneDP - https://arxiv.org/abs/2410.21257 (project: https://research.nvidia.com/labs/dir/onedp/)
- **Fast Generation from Diffusion Models:** NVIDIA FastGen
  Code: https://github.com/NVlabs/FastGen

**Serving-side (inference optimization):**
- **PyTorch-native and Flexible Inference Engine with Hybrid Cache Acceleration and Parallelism for DiTs:** Cache-DiT
  Code: https://github.com/vipshop/cache-dit
- **From Instantaneous to Average Velocity for Accelerating Flow Matching Inference:** MeanCache - https://arxiv.org/pdf/2601.19961
  Code: https://github.com/UnicomAI/MeanCache

### Step 7 - If you just need a faster expert, distill the VLA
- **Refined Policy Distillation (RL refinement to compact expert):** https://arxiv.org/abs/2503.05833

---

## 7) Minimal deployment checklist (refer this before you “start optimizing”)

1. **Choose runtime target per submodule** (vision / language / action head)  
2. Confirm **supported ops & shapes** (QNN/Core ML/TensorRT)  
3. Choose quantization scheme **that has kernels on your device**  
4. Validate in sim (LIBERO / RLBench / MimicGen / Isaac Lab) + replay logs  
5. Only then chase “fancy” pruning / parallel decoding papers

---

## 8) “Good defaults” for an on-device robotics loop

- Run the **policy** at 2–10 Hz (or slower), but run the **controller** at 50–200 Hz.  
- Use **action chunking** so one policy inference yields multiple fine-grained control steps.
- Use **async inference** (don’t block camera capture/control threads).

Pseudo-skeleton:
```python
# policy_rate << control_rate
while robot.is_running():
    obs = get_observation()  # camera + proprio
    if time_to_run_policy():
        chunk = policy(obs, instruction)   # e.g., predicts T future actions
    act = chunk.pop(0) if chunk else fallback_action()
    robot.step(act)  # high-rate control
```

---

## 9) Where this note connects in the repo

- For broader “SoC ML stacks” (training + inference, runtimes, compilers): see **ML Training & Inference on SoCs**.
- For model compression theory/practice (quant/prune/distill): see the **Optimizing Models** notes.

---
