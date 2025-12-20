# Model Optimization in 2025: A Survey

**An T. Le, Hanoi, Dec 2025**

Foundation models keep getting larger, context windows keep getting longer, and deployment constraints keep getting tighter. The result is that “model optimization” is no longer one trick, it is a stack of techniques that touch weights, activations, KV cache, kernels, runtime scheduling, and distributed training.

This post surveys the most used, SOTA model optimization methods for foundation models (LLMs, VLMs, diffusion), with a focus on what actually changes memory, latency, or throughput in real systems, and where you can find working code.

---

## The optimization stack

When you profile an LLM service, you usually see a handful of dominant cost centers:

* **Weights** (VRAM footprint, bandwidth during decoding)
* **KV cache** (VRAM footprint, bandwidth, fragmentation under multi-tenant serving)
* **Attention and GEMM kernels** (fusion, IO efficiency, precision)
* **Scheduling** (batching, prefill vs decode balance, cache reuse)
* **Training memory** (optimizer states, gradients, sharding, precision)

Everything below maps to one or more of those bottlenecks.

---

## 1. Quantization (largest memory win, usually best first move)

### Weight-only PTQ: keep activations high precision, quantize weights to 4-bit

#### [GPTQ](https://github.com/IST-DASLab/gptq)

GPTQ is a post-training quantization method that aims to preserve accuracy while compressing LLM weights, commonly to INT4. In practice, weight-only INT4 is often the easiest way to fit bigger models on the same GPU, and it frequently improves throughput because decoding is usually memory bandwidth bound.

A common extension path is experimenting with alternative packing and formats, for example [GPTQ plus GGUF “K-quants” work](https://github.com/IST-DASLab/gptq-gguf-toolkit).

#### [AWQ](https://github.com/mit-han-lab/llm-awq)

AWQ uses activation statistics to decide which weights or channels need extra protection, and is widely used for accurate INT4 and sometimes INT3 weight-only quantization. It is popular because it balances accuracy and hardware practicality.

**Discussion: when weight-only wins**

* Your bottleneck is **memory bandwidth** during autoregressive decoding.
* You want a **drop-in** accuracy preserving path, without engineering activation quantization.
* Your serving engine supports the exact packing format (this is the hidden make-or-break detail).

---

### W8A8 (INT8 weights and INT8 activations): higher performance, higher sensitivity

#### [SmoothQuant](https://github.com/mit-han-lab/smoothquant)

SmoothQuant is a training-free method designed to make W8A8 viable for large language models by handling activation outliers via a smoothing transformation. It is a common bridge to fast INT8 inference in vendor stacks.

**Discussion: why W8A8 is tricky**

* Activations have outliers that can dominate quantization error.
* Calibration data and layer-specific behavior matter more than weight-only.
* The speedups are best when your stack has excellent INT8 kernels and fusions.

---

### Practical k-bit infrastructure

#### [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)

bitsandbytes is a widely used PyTorch library enabling k-bit workflows for both inference and training. It is a common backbone for many practical quantization and adapter fine-tuning pipelines.

---

### 4-bit fine-tuning without full fine-tune cost

#### [QLoRA](https://github.com/artidoro/qlora)

QLoRA fine-tunes by backpropagating through a frozen 4-bit quantized base model into trainable low-rank adapters. The core value is that it makes high-quality adaptation feasible under tight VRAM budgets.

QLoRA is typically paired with:

* [PEFT](https://github.com/huggingface/peft) for adapter implementations and training ergonomics
* [LoRA](https://github.com/microsoft/LoRA) for the original reference implementation of loralib
* [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) for 4-bit storage and kernels

**Discussion: what you really get with QLoRA**

* You are optimizing the **iteration loop** more than the final inference speed.
* Deployment can be “serve with adapters” or “merge adapters,” each has different performance and operational tradeoffs.
* Quality depends more on data quality and adapter configuration than most people expect.

---

### Ultra-low-bit PTQ (2 to 4 bits): pushing compression harder

These are useful when INT4 is not enough, or you are targeting edge and want extreme memory reduction.

#### [AutoRound](https://github.com/intel/auto-round)

AutoRound is positioned as an advanced quantization toolkit for LLMs and VLMs, targeting 2 to 4-bit with minimal tuning. It is also notable for aiming at broad hardware compatibility.

#### [AQLM](https://github.com/Vahe1994/AQLM)

AQLM uses additive quantization concepts to push toward very low bits per parameter. This is often more complex than standard INT4, but can keep accuracy surprisingly strong at very aggressive compression levels when supported by the inference stack. 

#### [HQQ](https://github.com/dropbox/hqq)

HQQ is a fast quantizer that emphasizes skipping calibration data, which is handy when you do not have a representative calibration set or you want a rapid quantization workflow.

**Discussion: the reality of 2 to 3-bit**

* Accuracy can be very workload dependent, you should validate on your real prompts, not only perplexity.
* Kernel support and packing formats often decide if you get speedups or only memory savings.
* At very low bits, the “systems overhead” can become a bigger fraction of runtime.

---

## 2. KV cache optimization: the hidden bottleneck for long context and high concurrency

If you serve long prompts or many concurrent users, KV cache can dominate GPU memory and influence throughput via fragmentation and bandwidth.

#### [ZipCache](https://github.com/ThisisBillhe/ZipCache)

ZipCache targets KV cache quantization with mixed precision and salient token identification, aiming for accuracy while reducing memory and latency overhead.

#### [KVQuant](https://github.com/SqueezeAILab/KVQuant)

KVQuant focuses on enabling extremely long context inference via efficient KV cache quantization methodology, with a strong emphasis on long-context regimes.

**Discussion, when KV cache work matters most**

* Long context chat, heavy RAG prompts, tool traces, or multi-turn sessions.
* High concurrency, where KV cache memory is your capacity limiter.
* Systems that need predictable latency, because cache pressure can cause sharp performance cliffs.

---

## 3. Sparsity and pruning: powerful on paper, conditional in practice

Pruning can reduce compute and memory, but speedups depend heavily on kernel support and sparsity structure.

#### [SparseGPT](https://github.com/IST-DASLab/sparsegpt)

SparseGPT demonstrates one-shot pruning of large language models, including unstructured and n:m sparsity, and even sparse plus quantized configurations.

#### Structured 2:4 plus low-bit kernels: [Sparse-Marlin](https://github.com/IST-DASLab/Sparse-Marlin)

Sparse-Marlin extends a 4-bit kernel with support for 2:4 sparsity, which is one of the few sparsity patterns that can map well to modern GPU acceleration paths when the whole pipeline is aligned.

**Discussion: when sparsity is worth it**

* You can enforce a hardware-friendly pattern (often structured).
* You control kernels and deployment environment, not just the model checkpoint.
* You have benchmarks showing that sparse kernels win over dense INT4 for your shapes and batch regimes.

---

## 4. Decoding acceleration: reducing the number of expensive target-model steps

Autoregressive decoding is sequential, so methods that reduce target forward passes can yield big speedups.

#### [Speculative Decoding](https://github.com/romsto/Speculative-Decoding)

Speculative decoding uses a smaller “draft” model to propose tokens, then verifies them with the target model, preserving the output distribution while reducing serial decoding steps when acceptance rates are high.

#### Multi-head decoding: [Medusa](https://github.com/FasterDecoding/Medusa)

Medusa accelerates generation using multiple decoding heads, proposing multiple tokens per step and verifying efficiently.

**Discussion, when decoding tricks pay off**

* Your draft model is strong enough to yield high acceptance.
* You serve many “easy” tokens, like structured outputs, coding boilerplate, or repeated patterns.
* Your serving engine integrates the feature well, otherwise overhead can cancel gains.

---

## 5. Kernels and serving engines: where “real throughput” often comes from

You can have a well-quantized model and still get poor performance if the runtime cannot batch efficiently or manage KV cache well.

#### Attention kernels: [FlashAttention](https://github.com/Dao-AILab/flash-attention)

FlashAttention provides fast, memory-efficient exact attention kernels, including FlashAttention-2, and is commonly integrated into modern training and inference stacks.

#### Serving engine: [vLLM](https://github.com/vllm-project/vllm)

vLLM emphasizes throughput and memory efficiency with PagedAttention, continuous batching, and a growing set of quantization and decoding acceleration features. It is often the highest ROI change for production serving when GPU utilization is poor.

#### NVIDIA engine: [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

TensorRT-LLM is an open-source inference library focused on NVIDIA GPUs, including custom kernels, paged KV caching, multiple quantization paths, and speculative decoding support. It tends to shine when you want maximal performance and are willing to adopt an NVIDIA-centric build and deployment workflow.

**Discussion: a practical serving mindset**

* Optimize utilization first: batching, scheduling, KV management.
* Then optimize math: quantization, kernels, decoding shortcuts.
* Always validate quality under your exact sampler settings, temperature, top-p, and prompt distribution.

---

## 6. Training-scale optimization: making big runs feasible

Large foundation model training and large-scale fine-tuning are often memory-limited by optimizer state, gradients, and parameter replicas.

#### ZeRO and more: [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)

DeepSpeed provides ZeRO optimizations and broader distributed training and inference tooling for scaling large models efficiently.

Examples and recipes live in: [DeepSpeedExamples](https://github.com/deepspeedai/DeepSpeedExamples).

#### Native sharding: PyTorch FSDP

[PyTorch Fully Sharded Data Parallel (FSDP)](https://docs.pytorch.org/docs/stable/fsdp.html) is a core PyTorch approach for sharding parameters, gradients, and optimizer states across ranks. It is attractive when you want strong integration with the PyTorch ecosystem and fewer external dependencies.

#### FP8 building blocks: [TransformerEngine](https://github.com/NVIDIA/TransformerEngine)

Transformer Engine provides accelerated transformer components, including FP8 support on newer NVIDIA GPUs, aiming for higher throughput and lower memory in both training and inference.

**Discussion: picking a training stack**

* Choose based on integration and debugging comfort, not only theoretical performance.
* Communication patterns and cluster topology can dominate at scale.
* Precision changes (BF16, FP8) can require more careful stability and monitoring.

---

## 7. Diffusion and generative vision foundation models: fewer steps beats faster steps

For diffusion-style foundation models, the number of denoising steps is often the main latency driver.

#### Few-step diffusion: [Latent Consistency Models](https://github.com/luosiallen/latent-consistency-model)

LCM targets high-resolution generation with few-step inference by changing the sampling behavior, often delivering dramatic latency improvements compared to classic multi-step diffusion sampling.

#### Open model codebase hub: [Stability AI generative-models](https://github.com/Stability-AI/generative-models)

This repository is a central codebase for several Stability AI generative model releases and research-oriented implementations.

---

## 8. Putting it together: common model optimization stacks

### High-throughput LLM serving on NVIDIA GPUs

A commonly recipe looks like:

* Serve with [vLLM](https://github.com/vllm-project/vllm) for batching and KV management.
* Use [FlashAttention](https://github.com/Dao-AILab/flash-attention) if your model and GPU support it.
* Quantize weights with [AWQ](https://github.com/mit-han-lab/llm-awq) or [GPTQ](https://github.com/IST-DASLab/gptq).
* Add KV cache compression if long context is the limiter, for example [ZipCache](https://github.com/ThisisBillhe/ZipCache) or [KVQuant](https://github.com/SqueezeAILab/KVQuant).
* Consider decoding acceleration via [Medusa](https://github.com/FasterDecoding/Medusa) or speculative decoding if your prompts benefit.

### You want maximum NVIDIA performance and are willing to commit to the ecosystem

* Use [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) as the core engine
* Consider **FP8** paths via [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) for training and possibly inference depending on the engine path.
* Add 2:4 sparsity only if you commit to the [full pipeline](https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/) (pattern, training or pruning, kernel support)

### You need extreme compression or edge constraints

* Explore 2 to 4-bit PTQ via [AutoRound](https://github.com/intel/auto-round), [AQLM](https://github.com/Vahe1994/AQLM), or [HQQ](https://github.com/dropbox/hqq)
* If quality is still not enough, distill to a smaller student (then quantize again)
* Validate with real prompts, not just perplexity, because behavior changes at very low bit rates.

### If you deploy on edge or CPU heavy environments

* Distillation usually wins first, quantization second. A standard example is [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased/blame/3f08e1ec344e35f4999059e93ca45219abf01429/README.md).
* Quantization can be done with the smaller student using TorchAO or your target runtime.

