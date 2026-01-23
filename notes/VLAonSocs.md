# VLA on SoCs: Deployment Notes (Landscape & Practical)

**Ho T. Hung, An T. Le**

**Hanoi, Jan 2026**

*Scope:* **Vision-Language-Action (VLA)** *deployment* (inference-first) for robotics / embodied agents on **edge SoCs** (phones/tablets, laptops-as-edge, SBCs, Jetson-class modules).  
We covers: (1) model shapes (end-to-end VLA vs split VLM+policy), (2) optimization (quantization, pruning, distillation, cache/scheduling tricks), and (3) runtime lanes (Core ML/ANE, QNN, TensorRT, RKNN, LiteRT/ORT/OpenVINO) + profiling/packaging.  

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

## 1) Common VLA architectures

### A) Autoregressive VLM -> discrete action tokens (common)
**Examples:** OpenVLA, SmolVLA, many “token-based” VLAs  
- Encode image(s) -> visual tokens  
- Condition on instruction -> decode **action tokens** (often 7–DoF or chunked action sequences)
- Great for *generalization*, but can be **decode-bound** and **vision-token heavy**.

### B) VLM + diffusion / flow-matching action head (common for smooth actions)
- VLM provides task context; diffusion/flow head generates continuous actions.
- Often more sample efficient / smooth, but **multi-step inference** can be too slow unless distilled (see §6).

### C) Split pipeline (often the most SoC-friendly)
- **On-device perception** (small VLM/ViT) + **tiny policy/controller**
- Optional: off-device LLM “planner” (cloud or nearby GPU)  
- Sacrifices some end-to-end purity but is usually easier to ship.

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

## 3) Lane-specific toolchains (links you’ll actually use)

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


## 3.1 Common model artifacts (what you’ll actually ship)

When people say “deploy the VLA”, it often means producing multiple artifacts:

- **PyTorch checkpoint** (`.pt/.pth`) - research + baseline correctness
- **ONNX** (`.onnx`) - common interchange, especially for ORT / QNN EP
- **Core ML** (`.mlpackage` / `.mlmodel`) - iOS/macOS app bundle, ANE/GPU targeting
- **QNN / QAIRT context binaries** - Snapdragon NPU-deployable compiled assets
- **TensorRT engines** (`.plan`) - Jetson GPU-optimized engines (build on-device or in matching containers)
- **RKNN / RKLLM formats** - Rockchip-converted assets for NPU/runtime
- **GGUF / GGML-family** - CPU/GPU inference stacks like llama.cpp (useful for LLM-only subsystems)

Practical implication: plan for **multiple conversion + calibration passes**, and budget time for “op coverage” debugging.

---

## 3.2 Profiling & debugging (edge reality)

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

### 5.2 NanoVLA (routing + decoupling for edge)
- Paper: https://arxiv.org/abs/2510.25122  
- Claims focus on **late fusion** (decouple V/L), **routing**, and **chunking** to reduce edge cost.

---

### 5.3 TinyVLA (tiny-scale policy learning)
- Paper: https://arxiv.org/abs/2409.12514  
- Code: https://github.com/JayceWen/tinyvla

---

### 5.4 RoboMamba (state-space / Mamba-style efficiency)
- Paper: https://arxiv.org/abs/2406.04339  
- Project: https://robomamba.github.io/

---

### 5.5 MoLe‑VLA (dynamic layer skipping)
- Paper: https://arxiv.org/abs/2503.20384  
- Code: https://github.com/RoyZry98/MoLe-VLA-Pytorch

---

## 6) Optimization playbook (SoC-focused)

### Step 0 - Measure end-to-end (not just tokens/sec)
Track **control-loop metrics**:
- Policy latency (p50/p95), jitter
- Action update rate achieved on robot
- Closed-loop success rate (simulation + real)
- Power/thermals (sustained)

### Step 1 - Make the model “edge-shaped”
Low-risk levers:
- Reduce **image resolution** (e.g., $224^2$ -> $160^2$) *if success rate holds*
- Reduce **frame rate** and use **action chunking**
- Prefer **single image** or a small temporal window (avoid long video token streams)
- Enforce **static shapes** where your compiler needs it (QNN / TensorRT / Core ML)

### Step 2 - Quantize (usually the biggest memory win)
- LLM/VLM: int8 / int4 weights where kernels exist (platform-dependent)  
- Vision encoder: often int8-friendly; keep normalization consistent  
- Action head: fp16/bf16 often fine; quantize only if it’s a bottleneck

Apple note: Core ML tools explicitly supports **4-bit and 8-bit weight quantization** (and optional 8-bit activations).  
Qualcomm note: QNN compilation/EP often expects **quantized** graphs to reach NPU speedups.

### Step 3 - Cut visual tokens (often the biggest latency win)
Common patterns:
- Token pruning (static or per-layer)
- Token caching across frames (robotics has high temporal redundancy)
- Late fusion / decoupling so you can reuse parts

Representative papers (mostly training-free):
- **Token caching:** VLA‑Cache - https://arxiv.org/abs/2502.02175 (code: https://github.com/siyuhsu/vla-cache)
- **Dual-level pruning:** VLA‑Pruner - https://arxiv.org/abs/2511.16449  
- **Self-speculative pruning:** SpecPrune‑VLA - https://arxiv.org/abs/2509.05614  
- **Instruction-guided token compression:** Compressor‑VLA - https://arxiv.org/abs/2511.18950  
- **Driving-focused pruning:** FastDriveVLA - https://arxiv.org/abs/2507.23318

### Step 4 - Make action tokens efficient (quality + speed)
If action discretization is hurting dexterity or sequence length:
- **FAST / FAST+ action tokenization:** https://arxiv.org/abs/2501.09747  
- **Vector-quantized action tokenizers:** VQ‑VLA - https://arxiv.org/abs/2507.01016 (code: https://github.com/xiaoxiao0406/VQ-VLA)

### Step 5 - Reduce decoding overhead
- **Parallel decoding for action chunking:** PD‑VLA - https://arxiv.org/abs/2503.02310  
- **Early exit decoding / consistency-style:** CEED‑VLA - https://arxiv.org/abs/2506.13725

### Step 6 - If you have diffusion policies, distill them
Multi-step diffusion is often the blocker for real-time on SoCs.
- **One-step diffusion distillation:** OneDP - https://arxiv.org/abs/2410.21257 (project: https://research.nvidia.com/labs/dir/onedp/)

### Step 7 - If you just need a faster expert, distill the VLA
- **Refined Policy Distillation (RL refinement to compact expert):** https://arxiv.org/abs/2503.05833

---

## 7) Minimal deployment checklist (use this before you “optimize”)

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
