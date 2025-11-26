# Optimizing Models: A Train Of Thought

**An T. Le, Hanoi, Nov 2025**

In practice, modern foundation models are optimized in two layers:

- (A) Model-level optimization: techniques such as quantization, pruning and sparsity, knowledge distillation, and low-rank / factorized parameterization that reduce compute and memory while preserving task performance.
- (B) Edge-level deployment optimization: toolchains and runtimes (e.g., TensorRT / ModelOpt, Intel Neural Compressor + OpenVINO, TorchAO, ONNX Runtime, TFLite, TVM, ncnn, etc.) that take an optimized model and generate hardware-specific inference engines for GPUs, CPUs, and embedded/edge SoCs.

Let's dive into mainstream pathways that smashes foundation models then optimizes for deployments on SoCs.

---

## 1. Mainstream **(A) Model-level optimization**

### 1.1 Quantization (almost always the first step)

Goal: reduce weights/activations from FP16/FP32 -> INT8/INT4/FP8/NVFP4 while keeping accuracy.

Common variants (esp. for big transformers & FMs) ([see Model Opt Survey][1]):

* **Post-Training Quantization (PTQ)**

  * No retraining; use a small calibration set.
  * Static vs dynamic:

    * *Static:* collect activation stats on a calibration set, quantize both weights + activations (INT8 is standard).
    * *Dynamic:* quantize weights offline, activations quantized on-the-fly (common for CPUs).
  * Foundation-model-specific flavors:

    * **SmoothQuant**, **AWQ (Activation-aware Weight Quantization)**, “AutoQuantize” for robust 8/4-bit on transformers. ([see Edge AI and Vision Alliance][2])
    * KV-cache-only quantization (4–8 bit) to shrink memory and speed long context inference.

* **Quantization-Aware Training (QAT)**

  * Simulate quantization during finetuning; highest accuracy, but needs training loop.
  * Used when you need INT8/INT4 on tough tasks (manipulation, safety-critical robotics) and PTQ alone is too lossy. ([QAT][3])

* **Mixed precision & custom formats**

  * FP16 / BF16 for most layers, FP32 on layernorms / critical parts.
  * NVIDIA-specific: **FP8** and **NVFP4** (4-bit) formats, integrated in TensorRT-LLM / ModelOpt optimized checkpoints. ([see Edge AI and Vision Alliance][2])

**Example:** for GR00T, a very typical recipe is:

* Vision + language backbones: 8-bit (or FP8/NVFP4)
* Action/diffusion head + final layers: stay FP16 or BF16

---

### 1.2 Pruning & sparsity

Goal: remove “unimportant” parameters, often combined with quant.

Main flavors for large transformers & FMs ([see docs][4]):

* **Unstructured pruning (weight sparsity)**

  * Magnitude pruning: drop smallest-magnitude weights.
  * Activation-based pruning: use calibration activations to decide importance.
  * Easy to apply, but needs sparse-aware kernels to really speed up.

* **Structured pruning (real-world speedups)**

  * Prune **attention heads**, **MLP channels**, **entire blocks/layers**, or **tokens**.
  * For transformers: common to prune least-useful heads, shrink FFN hidden dims, or drop shallow/deep blocks.

* **N:M sparsity / semi-structured sparsity**

  * E.g. 2:4 sparsity (2 non-zeros per 4 weights) that matches hardware support on NVIDIA GPUs.
  * Supported in some vendor libraries (TensorRT, cuSparse, TorchAO) to get actual speedups. ([GitHub][5])

**Example:** for GR00T, a very typical recipe is:

* Prune *select layers/heads* in the language backbone and MLPs,
* Maybe leave vision encoder and action head less pruned because they’re already critical & smaller.

---

### 1.3 Knowledge Distillation (KD)

Goal: train a smaller/cheaper “student” to mimic a large “teacher”. ([see PyTorch documentation][6])

Main flavors:

* **Logit distillation:** student matches teacher’s soft logits (standard Hinton KD).
* **Feature distillation:** align hidden states / attention maps; popular for transformers.
* **Sequence / behavior-level distillation:** student imitates teacher’s generated trajectories / actions.

**Example**: for robotics FMs:

* **Policy distillation:** teacher GR00T-like model generates actions, student learns a simpler policy (smaller transformer or MLP head).
* **Task-specific students:** e.g., a small arm-only policy distilled from a huge general VLA, or a lower-frequency high-level planner distilled into a compact controller.

KD is often combined with pruning/quant:

> prune -> finetune with distillation -> quantize -> (optional) KD again to recover accuracy.

---

### 1.4 Low-rank & factorization tricks

Often grouped with compression even if they’re also “adaptation” methods ([see Model Optimization Survey][1]):

* **LoRA / low-rank adapters:** represent deltas in low rank; you can freeze the base and only train low-rank matrices, then optionally *merge* or keep them separate.
* **Tensor decompositions:** factor big weight matrices (e.g., SVD, Tucker, CP) to smaller components.
* **Block sharing / parameter tying:** reuse blocks across layers.

These are widely used to *reduce trainable parameters* and sometimes inference cost, especially if merged back into a smaller dense matrix.

---

## 2. Mainstream **(B) Edge-level pipelines / toolchains**

### 2.1 NVIDIA-centric: TensorRT-LLM + TensorRT Model Optimizer

This toolchain is popular recently (2024–2025), especially for large transformers and FMs. For GR00T-style models on NVIDIA GPUs, this is the canonical stack.

* **TensorRT Model Optimizer (ModelOpt)**: library of SOTA optimization techniques:
  quantization, distillation, pruning, speculative decoding, sparsity, etc. ([GitHub][5])
* **Pipeline sketch:**

  1. Start from PyTorch / HuggingFace checkpoint (e.g., GR00T N1.5).
  2. Export to ONNX or use TensorRT-LLM’s direct integration.
  3. Run **PTQ or QAT** with ModelOpt (INT8 / FP8 / NVFP4), optionally with AWQ/SmoothQuant-style calibration.
  4. Apply **pruning** (magnitude/activation, structured heads/MLP) if needed.
  5. Optionally run **distillation** (ModelOpt has KD utilities for LLMs) to recover accuracy in a smaller student.
  6. Build a **TensorRT engine** and deploy (Jetson, DGX, Thor, etc).

NVIDIA is already shipping *pre-quantized, ModelOpt-optimized* checkpoints for many LLMs and generative models, and similar recipes are intended for GR00T-class models. ([see Nvidia docs][7])

---

### 2.2 Intel / CPU-oriented: Intel Neural Compressor + OpenVINO

If you care about x86 / CPU inference (servers, industrial PCs), the mainstream is:

* **Intel Neural Compressor (INC)**: one library that does **quantization, pruning, distillation, and even NAS**, across PyTorch / TF / ONNX. ([INC][8])
* Typical pipeline:

  1. Start with PyTorch model.
  2. Wrap it in INC’s **INCQuantizer**.
  3. Let INC search over quantization/pruning configs with accuracy constraints.
  4. Optionally enable KD from a larger teacher.
  5. Export to **OpenVINO IR** for deployment.

There are HF tutorials doing exactly this for transformer models (DistilBERT etc.). ([see this][9])

---

### 2.3 PyTorch-native: TorchAO + built-in PyTorch APIs

For research & quick iteration where you want to stay in pure PyTorch:

* **TorchAO (Training-to-Serving Model Optimization)**
  A PyTorch-native framework that unifies **quantization + sparsity** across training and serving, aimed at large models. ([see paper][10])

* Plus standard PyTorch utilities:

  * `torch.quantization` / `torch.ao.quantization` for PTQ + QAT. ([see PyTorch docs][11])
  * `torch.nn.utils.prune` for various pruning schemes. ([see PyTorch docs][12])
  * KD implemented manually or via common recipes (e.g., aligning logits/hidden states). ([see PyTorch docs][6])

Typical flow:

1. Prototype pruning/quant (TorchAO / PyTorch).
2. Train / distill student.
3. Optionally export to ONNX and then to a deployment runtime (TensorRT, ONNX Runtime, TVM, etc.).

---

### 2.4 Framework-agnostic / edge-oriented runtimes

For ARM SoCs, MCUs, or heterogeneous edge boxes (i.MX, AM69, Jetson Nano/Orin, etc.), mainstream choices are:

* **ONNX Runtime** with built-in PTQ + graph optimizations (fusion, constant folding, etc.). ([see this][13])
* **TensorFlow Lite** (especially mobile).
* **TVM / Apache TVM**: auto-tuned kernels, supports quantization and some pruning-aware compilation.
* **ncnn**: lightweight C++ runtime with its own quantization and graph optimizations (popular on mobile / embedded).

These generally expect:

1. You do pruning/KD in PyTorch/TF.
2. Export ONNX / TFLite.
3. Run their **quantization + graph optimization** passes.
4. Deploy as a small runtime binary.

---

### 2.5 “All-in-one” compression frameworks

Besides vendor stacks, a few general frameworks are used a lot for *automation*:

* **Intel Neural Compressor** again (does multi-framework automation). ([GitHub][14])
* **DeepSpeed** compression modules for LLM sparsity and quantization. ([Deepchecks][15])
* **SparseML / NeuralMagic**, low-rank / sparse finetuning tools, etc.
* Commercial tools like [Pruna.ai](https://www.pruna.ai/), which wrap search over pruning/quantization strategies around PyTorch/ONNX models.

These usually provide:

* Config-driven definitions of which layers to quantize/prune,
* Automatic search over bitwidths/sparsity,
* Optional KD.

---

## 3. Example: GR00T-like robotics FM

For a GR00T-style VLA you can try combining:

1. **Baseline profiling** on your target (Jetson, Thor, TI SK-AM69, i.MX93, etc.).
2. **PTQ 8-bit / FP8** on most transformer layers (vision + language backbones), keep diffusion/action heads in higher precision.
3. **Structured pruning** of attention heads & MLP channels in the language trunk; maybe light pruning in vision backbone.
4. **Distillation** to:

   * a smaller VLA (fewer layers/heads), or
   * task-specific students (e.g., just a manipulator policy) if you only need a subset of GR00T’s capabilities.
5. Export to your deployment stack:

   * NVIDIA path: PyTorch -> ModelOpt (quant+prune+KD) -> TensorRT engine.
   * CPU / non-NVIDIA path: PyTorch -> INC / ONNX Runtime / TVM / ncnn with their PTQ flows (This maybe possible for some SoCs depending on computing & memory & DSP capacity.)

---

## 4. Discussions

In model optimization, you treat model as an algorithmic object: a graph with parameters, structure, and training data. The goal is to **reduce compute, memory, and latency while preserving (or sometimes trading off) task performance**.

### 4.1 Parameter/computation reduction

* **Quantization**
  Lower the numeric precision of weights/activations (FP32 -> FP16/BF16 -> INT8/FP8/4-bit).

  * *Benefits*: big cuts in memory bandwidth and throughput, often with negligible loss under good calibration or QAT.
  * *Variants*: post-training quantization, quantization-aware training, mixed precision (critical parts stay higher precision).

* **Pruning & sparsity**
  Remove parameters or constrain them to sparse patterns.

  * *Unstructured*: remove individual small-magnitude weights -> high compression, but needs sparse kernels to see speedups.
  * *Structured*: drop whole channels, heads, blocks -> easier to accelerate and reason about.
  * *Hardware-aligned*: N:M sparsity patterns that specific accelerators exploit.

* **Low-rank / factorization**
  Factor large matrices into low-rank pieces, or use adapters (LoRA-style) that reduce effective parameter count.

  * Especially useful when you want *both* efficient finetuning and some inference savings (if you merge factors).

### 4.2 Knowledge & behavior preservation

Once you’ve made the model “smaller” or “cheaper,” you need to keep it smart:

* **Knowledge distillation**
  Train a smaller “student” to mimic a larger “teacher” via logits, features, or behaviors.

  * Classic KD for classification/sequence tasks.
  * Policy/behavior distillation for robotics, recommender systems, etc.

* **Task-aware finetuning**
  After quant/prune, you often do a short finetune (sometimes with KD) on the key downstream tasks to recover lost performance.

### 4.3 Architectural & training tricks

* **Architecture search & simplification**: choosing more hardware-friendly block structures (e.g., fewer attention heads, smaller FFNs, more depth-wise separable convs, etc.).
* **Curriculum & multi-task strategies**: shaping how the model learns so it can later be compressed more aggressively without collapsing.
* **Regularization for compressibility**: encouraging sparsity or low-rank structure during training so later pruning/quantization is easier.

You can think of this layer as:

> *“Given many hardware choices later, how do I make the core model inherently efficient and compressible?”*

---

Once you have a “compressed” or well-designed model, you still need to run it efficiently on **specific hardware**: GPUs, CPUs, NPUs, ASICs, microcontrollers, etc. This is where **compilers, runtimes, and system-level tricks** come in.

### 4.4 Graph-level optimization & compilation

* **Operator fusion & graph rewriting**: Combine sequences of ops into single kernels, fold constants, pre-compute static parts.
* **Layout & scheduling**: Choose memory layouts (NHWC vs NCHW, packed vs planar), tiling strategies, and stream scheduling to match caches and compute units.
* **Backend selection**: Decide per-op implementation: CUDA vs TensorRT kernels vs cuDNN vs custom kernels, or CPU vs GPU vs NPU placement.

### 4.5 Hardware-aware quantization & formats

Even if the model “supports INT8,” each hardware stack has its own preferred formats:

* GPUs may prefer FP16/FP8/NVFP4;
* CPUs often use INT8 with specific instruction sets;
* Tiny MCUs use per-channel INT8 or INT4 with strict memory limits.

Edge-level toolchains take your **abstract quantization plan** (e.g., “these layers 8-bit, those 16-bit”) and map it to **actual kernels and calibration procedures** that match the device.

### 4.6 Runtime orchestration

* **Batching & concurrency**: trade off latency vs throughput, share accelerators across multiple models or clients.
* **Memory management**: activation checkpointing, KV-cache management, overlapping copy+compute.
* **Fallback paths**: when a kernel isn’t supported on a given accelerator, gracefully fall back to another device or a slower implementation.

You can think of this layer as:

> *“Given this fixed model, how do I turn it into a hardware-specific engine that squeezes every millisecond and watt on the target device?”*

---

## 5. How the two layers interact

In reality, these aren’t sequential, they’re a **feedback loop**:

1. **Profile on a target** -> identify slow/energy-hungry parts.
2. **Model-level changes** (quantize/prune/distill/architectural tweaks) targeting those hotspots.
3. **Recompile & redeploy** with hardware-specific toolchains.
4. **Measure again** -> iterate.

A few key patterns:

* **Co-design**:
  You don’t just compress blindly. You compress *where the hardware benefits most* (e.g., prune MLP channels on GPU, but keep convolutions dense if they’re already well-optimized).

* **Multi-target thinking**:
  One “model-level family” with multiple “deployment variants”:

  * High-precision / large model for cloud,
  * Aggressively quantized / distilled variant for edge.

* **Task-aware metrics**:
  It’s not enough to only track FLOPs/latency; you track **task-level degradation** (success rate, safety, robustness) and treat those as constraints in your optimization loop.

---

## 6. Concluding thoughts

* **Model optimization and edge optimization are inseparable.**
  A beautifully compressed model that doesn’t match hardware capabilities can be slower than a larger but hardware-friendly one. Conversely, a sophisticated compiler can’t save an over-sized, poorly structured model.

* **The “mainstream toolbox” is converging.**
  Across vision, language, and robotics, everyone is reaching for the same core tools—quantization, pruning/sparsity, distillation, low-rank tricks—then handing the result to increasingly capable compilers/runtimes.

* **The frontier is in *joint* design.**
  The most compelling work now treats:

  * model architecture,
  * training strategy,
  * compression plan, and
  * deployment stack
    as a single co-designed system, rather than separate phases.

* **For robotics and other real-world systems**, the optimization objective is not just “tokens per second” or “images per second,” but **safe, robust behavior under tight compute and power budgets**. That makes the two-layer view, *first shape the model, then specialize it to hardware, then iterate*, a practical framework for building foundation-model-driven systems that can actually leave the lab and live on edge devices.


[1]: https://aclanthology.org/2024.tacl-1.85.pdf "A Survey on Model Compression for Large Language Models"
[2]: https://www.edge-ai-vision.com/2025/08/optimizing-llms-for-performance-and-accuracy-with-post-training-quantization/ "Optimizing LLMs for Performance and Accuracy with Post- ..."
[3]: https://nvidia.github.io/TensorRT-Model-Optimizer/ "Welcome to Model Optimizer (ModelOpt) documentation!"
[4]: https://developer.nvidia.com/blog/pruning-and-distilling-llms-using-nvidia-tensorrt-model-optimizer/ "Pruning and Distilling LLMs Using NVIDIA TensorRT ..."
[5]: https://github.com/NVIDIA/TensorRT-Model-Optimizer "NVIDIA/TensorRT-Model-Optimizer"
[6]: https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html "Knowledge Distillation Tutorial"
[7]: https://developer.nvidia.com/blog/nvidia-hardware-innovations-and-open-source-contributions-are-shaping-ai/ "NVIDIA Hardware Innovations and Open Source ..."
[8]: https://www.intel.com/content/www/us/en/developer/articles/technical/an-easy-introduction-to-intel-neural-compressor.html "An Easy Introduction to Intel® Neural Compressor"
[9]: https://think.in2p3.fr/2023/06/14/compressing-the-transformer-optimization-of-distilbert-with-the-intel-neural-compressor/ "Optimization of DistilBERT with the Intel® Neural Compressor"
[10]: https://arxiv.org/abs/2507.16099 "PyTorch-Native Training-to-Serving Model Optimization"
[11]: https://docs.pytorch.org/tutorials/deep-dive.html "Deep Dive — PyTorch Tutorials 2.9.0+cu128 documentation"
[12]: https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html "Pruning Tutorial"
[13]: https://medium.com/%40akankshasinha247/model-compression-quantization-distillation-b5006cf41546 "Model Compression, Quantization & Distillation"
[14]: https://github.com/intel/neural-compressor "intel/neural-compressor: SOTA low-bit LLM quantization ..."
[15]: https://www.deepchecks.com/llm-pruning-and-distillation-importance/ "The Need for LLM Pruning and Distillation"
