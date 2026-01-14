# Model Optimization with Pruna.ai

**An T. Le, Hanoi, Jan 2026**

> TL;DR: **Pruna = one API to stack multiple model-level optimizations** (quantization, pruning, caching, compilation, kernel swaps), **plus built-in evaluation + save/load + deployment integrations**.

---

## 0) When Pruna is a good fit

Use Pruna when you want **developer-friendly, stackable optimizations** on **PyTorch / Hugging Face models** (LLMs, diffusion / flow-matching image/video models, ViTs, Whisper, etc.) with **fast iteration** and **built-in evaluation + reproducible configs**.  
Good especially if you want to **combine** techniques (e.g., *quantize + compile*, *cache + compile*, *kernel + quantize*).

Pruna is **not** an inference engine by itself; it often pairs best with deployment runtimes like **vLLM** (LLMs) or **Triton** (general serving), after you “smash” the model.

**Key concept:** **“Smashing”** = applying a `SmashConfig` via `smash(...)` to produce a `PrunaModel` (or a wrapped model object) with chosen optimizations.

---

## 1) Install + minimal setup

### Install (v0.3.0 in docs)
- Base install:
  - `pip install pruna==0.3.0`
- Install **all algorithms**:
  - `pip install "pruna[full]"`

### Extras you’ll likely want
- `stable-fast` (diffusion compiler path): `pip install "pruna[stable-fast]"`
- `gptq` (GPTQ quantization): `pip install "pruna[gptq]"`  
  (Pruna’s docs mention an extra index URL for some GPTQ dependencies—see the **Algorithms Overview** entry for `gptq`.)

**Docs:**  
- Install guide: https://docs.pruna.ai/en/stable/setup/install.html  
- Algorithms + per-algorithm extra deps: https://docs.pruna.ai/en/stable/compression.html  

---

## 2) Core workflow (repeatable + measurable)

### 2.1 Smash (optimize)
```python
from pruna import SmashConfig, smash

smash_config = SmashConfig()
smash_config["compiler"] = "torch_compile"
smash_config["quantizer"] = "hqq"  # example

optimized = smash(model=base_model, smash_config=smash_config)
```

**Docs:**  
- Smash your first model: https://docs.pruna.ai/en/stable/docs_pruna/user_manual/smash.html  
- SmashConfig (concepts + fields): https://docs.pruna.ai/en/stable/docs_pruna/user_manual/configure.html  
- `smash` API reference: https://docs.pruna.ai/en/stable/reference/smash.html  

### 2.2 Evaluate quality + perf (don’t skip this)
Pruna’s **EvaluationAgent + Task + PrunaDataModule** gives you a quick harness to compare **quality metrics** (task-specific) and **efficiency metrics** (latency, memory, etc.).

```python
import copy
from pruna import SmashConfig, smash
from pruna.data.pruna_datamodule import PrunaDataModule
from pruna.evaluation.evaluation_agent import EvaluationAgent
from pruna.evaluation.task import Task

# Base + smashed
base = pipe
smashed = smash(copy.deepcopy(pipe), SmashConfig(["deepcache"]))

task = Task(
    request=["clip_score", "psnr"],             # pick metrics
    datamodule=PrunaDataModule.from_string("LAION256"),
    device="cuda",
)
agent = EvaluationAgent(task)

base_res   = agent.evaluate(base)
smashed_res = agent.evaluate(smashed)
print(base_res.results, smashed_res.results)
```

**Docs:**  
- Evaluation overview: https://docs.pruna.ai/en/stable/reference/evaluation.html  
- Evaluation Agent guide: https://docs.pruna.ai/en/stable/docs_pruna/user_manual/evaluate.html  
- Example tutorial (CMMD): https://docs.pruna.ai/en/stable/docs_pruna/tutorials/evaluation_agent_cmmd.html  

### 2.3 Save/load + share
Pruna uses a Hugging Face–style workflow (`save_pretrained`, `push_to_hub`, `from_pretrained`) for smashed models.

```python
optimized.save_pretrained("saved_model/")
# or: optimized.push_to_hub("YourOrg/your-model-smashed")

from pruna import PrunaModel
loaded = PrunaModel.from_pretrained("saved_model/")
```

**Docs:**  
- Save & load: https://docs.pruna.ai/en/stable/docs_pruna/user_manual/save_load.html  

### 2.4 Deploy (common patterns)
Pruna includes guides/integrations for:
- Docker
- ComfyUI
- Triton Inference Server
- vLLM
- Replicate, Koyeb, AWS AMI, Lightning (LitServe)

**Docs:**  
- Integrations index: https://docs.pruna.ai/en/stable/setup/integrations.html  
- Triton: https://docs.pruna.ai/en/stable/setup/tritonserver.html  
- vLLM: https://docs.pruna.ai/en/stable/setup/vllm.html  

---

## 3) What Pruna can “smash” (model types)

From Pruna’s public package description / README, Pruna targets **multiple model families**, notably:
- **LLMs / transformer decoders** (CausalLM, “reasoning LLMs” in tutorials)
- **Diffusion + flow-matching image models** (incl. caching + stable-fast style compilation)
- **Video generation models** (tutorials)
- **Vision Transformers**
- **Speech recognition** (Whisper; IFW / WhisperS2T tooling)
- And my attempt to further smash [SmolVLA](../examples/pruna-smolvla/) :)

Quick entry points:
- Pruna GitHub: https://github.com/PrunaAI/pruna  
- Pruna docs tutorials: https://docs.pruna.ai/en/stable/docs_pruna/tutorials/index.html  
- PrunaAI Hugging Face org (pre-smashed models): https://huggingface.co/PrunaAI  

---

## 4) Optimization methods & tools inside Pruna (by category)

> The most “Pruna-ish” thing you can do is **stack** these with a single config, then **measure** with EvaluationAgent.

### 4.1 Batching (mostly Whisper)
- `ifw` (**Insanely Fast Whisper**) — speed/throughput via batching + low-level optimizations
- `whisper_s2t` — optimized Whisper speech-to-text pipeline

Where it shines:
- Transcription throughput (batching matters a lot).
Gotcha:
- Some batchers “prepare the model for inference with the batch size specified in the smash config” → set batch size to match deployment needs.

Docs: https://docs.pruna.ai/en/stable/compression.html

---

### 4.2 Caching (diffusion-style speedups)
Caching trades **some fidelity** for **fewer expensive ops** during iterative generation.

Algorithms in stable docs:
- `deepcache` — reuse U-Net features across steps (diffusion)
- `fastercache` — reuse attention states + skip unconditional branch compute (diffusion transformers)
- `fora` — reuse transformer block outputs for N steps, with a “start step” knob
- `pab` — skip attention compute between steps, reuse cached attention

Where it shines:
- Diffusion pipelines where the main loop repeats 20–50+ steps.
Gotchas:
- Quality drift varies by prompt + sampler; evaluate with your metrics (CLIP/CMMD/etc).

Docs: https://docs.pruna.ai/en/stable/compression.html

---

### 4.3 Compilation (graph lowering / runtime swaps)
Compilation aims for **lower latency** by compiling or switching runtimes.

Algorithms (stable docs):
- `torch_compile` — wraps `torch.compile` backends (CPU/CUDA), broad applicability
- `stable_fast` — diffusion-specific compiler path (kernel fusion + TorchScript graph)
- `c_generate`, `c_translate`, `c_whisper` — custom runtime path for generation/translation/Whisper (CUDA)

Where it shines:
- Stable workloads (fixed-ish shapes), repeated calls, production inference.
Gotchas:
- **Warm-up matters** (compile overhead). Benchmark after warm-up.
- Some compilation paths may be less portable or backend-dependent.

Docs: https://docs.pruna.ai/en/stable/compression.html

---

### 4.4 Kernel swaps (LLM attention)
- `flash_attn3` — FlashAttention 3 style attention kernel swap (CUDA)

Where it shines:
- Decoder LLM attention speed/memory improvements.
Gotchas:
- Kernel availability depends on GPU arch + build; watch for CUDA/PyTorch compatibility.

Docs: https://docs.pruna.ai/en/stable/compression.html

---

### 4.5 Factorization (diffusion transformers)
- `qkv_diffusers` — factorization for diffusers attention projections (Q/K/V), can reduce compute.

Where it shines:
- Diffusers transformer blocks.
Gotchas:
- Often interacts with other diffusion optimizations; rely on Pruna’s compatibility notes.

Docs: https://docs.pruna.ai/en/stable/compression.html

---

### 4.6 Pruning (parameter removal / sparsity)
Algorithms (stable docs):
- `torch_structured` — structured pruning
- `torch_unstructured` — unstructured pruning

Where it shines:
- When you can exploit sparsity in your runtime/hardware, or when pruning is paired with later compilation / distillation / recovery.
Gotchas:
- Real speedups depend heavily on *hardware + kernel support*; a sparse model isn’t automatically faster.
- Pruning often needs a dataset (or at least representative samples) for better results.

Docs: https://docs.pruna.ai/en/stable/compression.html

---

### 4.7 Quantization (the biggest lever for LLM memory)
Algorithms visible in stable docs (varies by model type):
- LLM-oriented:
  - `llm_int8` (weight-bit configuration)
  - `gptq`
  - `hqq`
  - `torchao`
- Diffusion-oriented:
  - `diffusers_int8`
  - `hqq_diffusers`
  - `quanto`
- Utility:
  - `half` (FP16-ish)

Where it shines:
- **LLMs:** memory footprint + throughput (esp. 8-bit / 4-bit weight-only)
- **Diffusion:** speed + VRAM (depends on pipeline)
Gotchas:
- Quantization can introduce quality regressions; always evaluate on your prompts/tasks.
- Some quantizers require extra dependencies or specific backends.

Docs: https://docs.pruna.ai/en/stable/compression.html

---

## 5) High-leverage “recipes” (what to try first)

### 5.1 LLMs
Start simple:
1) Quantize (4-bit/8-bit)  
2) Add `torch_compile` and/or `flash_attn3` if supported  
3) Serve with vLLM

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from pruna import SmashConfig, smash

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
mdl = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda")

cfg = SmashConfig()
cfg["quantizer"] = "llm_int8"  # or "gptq" / "hqq" / "torchao"
cfg["llm_int8_weight_bits"] = 4
cfg["compiler"] = "torch_compile"

opt = smash(mdl, cfg)
opt.save_pretrained("llama3-8b-smashed/")
```

Serve with vLLM:
- vLLM guide: https://docs.pruna.ai/en/stable/setup/vllm.html

**Compatibility note:** Pruna states full vLLM compatibility is “coming soon”, but you can already load Pruna-optimized models with supported quantizers like AutoAWQ, BitsAndBytes, GPTQ, TorchAO.  
See: https://www.pruna.ai/product-pages/compatibility-layer

---

### 5.2 Diffusion / image generation
Typical fast stack:
- `deepcache` (or other cacher) + `stable_fast` + (optional) `quanto` / `diffusers_int8`

```python
from diffusers import DiffusionPipeline
from pruna import SmashConfig, smash

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1").to("cuda")

cfg = SmashConfig(["deepcache", "stable_fast"])  # order can matter
opt_pipe = smash(pipe, cfg)
```

Evaluate:
- Image generation tutorial: https://docs.pruna.ai/en/stable/docs_pruna/tutorials/image_generation.html

---

### 5.3 Whisper / speech
Try `ifw` or `whisper_s2t`, but **set batch size** to match your target throughput.

Docs: https://docs.pruna.ai/en/stable/compression.html  
Tutorial: https://docs.pruna.ai/en/stable/docs_pruna/tutorials/index.html (Whisper tutorial)

---

## 6) Pruna Pro (paid): what you gain

> Disclaimer: I collect infos but haven't tried this mode!

Pruna Pro keeps the same interface, but adds:
- **OptimizationAgent**: automatically searches for the best `SmashConfig` given your objectives
- **Quality recovery algorithms**: recover accuracy after heavy compression (e.g., 4-bit)
- Additional proprietary / premium algorithms + priority support

Docs:  
- Transition to Pro: https://docs.pruna.ai/en/stable/docs_pruna_pro/user_manual/pruna_pro.html  
- Optimization Agent (older docs page, still useful for concepts): https://docs.pruna.ai/en/v0.2.3/docs_pruna_pro/user_manual/optimization_agent.html  

---

## 7) Caveats / gotchas (practical)

### 7.1 “Speedup” is not guaranteed
- **Unstructured sparsity** often doesn’t speed up unless your runtime has sparse kernels.
- Some caching methods speed up *sampling loops* but change the output distribution (quality regression risk).
- Compilation can pay off only after warm-up, and may be sensitive to shapes.

### 7.2 Compatibility constraints are real
Pruna’s algorithm entries list:
- supported devices (CPU / CUDA / distributed)
- required components (tokenizer, processor, dataset)
- compatibility with other algorithms

Use that as the source of truth: https://docs.pruna.ai/en/stable/compression.html

### 7.3 Benchmarking pitfalls
- Always separate **compile time** vs **steady-state inference**.
- Compare **latency** and **throughput**; they can move in opposite directions.
- Use representative inputs (seq length for LLMs, prompt styles for diffusion, etc.).

### 7.4 “Distillation” status (watch this)
- Pruna’s README/PyPI description mentions distillation as a supported technique.
- Older Pruna docs (e.g., v0.2.x) explicitly listed “Distillers” in the algorithm catalog.
- The **current stable algorithm catalog page** focuses on batchers/cachers/compilers/quantizers/pruners/etc. If you need distillation, check:
  - whether it moved to **Pruna Pro**, or
  - whether it’s temporarily removed/renamed in the current stable line.

Tip: treat distillation as “optional” until you see a concrete distiller in the Algorithms Overview for your version.

---

## 8) Comparison: Pruna vs other "end-to-end" optimization stacks

### How to choose (rule of thumb)
- **You want easy stacking across many PyTorch model families + built-in evaluation:** **Pruna**
- **You deploy on NVIDIA GPUs and want the tightest TensorRT/TensorRT-LLM path:** **NVIDIA ModelOpt**
- **You deploy on Intel CPUs/GPUs and want accuracy-driven quantization/pruning pipelines:** **Intel Neural Compressor**
- **You want OpenVINO IR + compression (PTQ/QAT) with strong Intel runtime integration:** **OpenVINO NNCF**
- **You want a hardware-aware ONNX optimization pipeline (conversion + optimization + tuning):** **Microsoft Olive**
- **You live in Hugging Face and want exporter + backend-specific optimization tooling:** **Hugging Face Optimum**
- **You want a PyTorch-native quantization/sparsity component (not a full pipeline):** **TorchAO**
- **You mainly care about LLM quant + sparsity for vLLM serving:** **LLM Compressor**

### Quick matrix (very compressed)
| Stack | “End-to-end”? | Sweet spot | Techniques | Notes |
|---|---:|---|---|---|
| **Pruna / Pruna Pro** | ✅ | PyTorch/HF models, stacking optimizations + eval | quant, prune, caching, compile, kernels (+ Pro agent/recovery) | Great UX; check per-algo compatibility + backend requirements. |
| **NVIDIA ModelOpt** | ✅ | NVIDIA GPU inference | quant, prune, distill, sparsity, speculative decoding | Integrates with TensorRT(-LLM), vLLM, etc. |
| **Intel Neural Compressor** | ✅ | Intel CPU/GPU | quant, prune, distill, NAS/tuning | Accuracy-driven tuning; multi-framework. |
| **OpenVINO NNCF** | ✅ | OpenVINO deployment | PTQ/QAT, sparsity, pruning | Produces OpenVINO-friendly optimized models; strong Intel runtime tie-in. |
| **Microsoft Olive** | ✅ | ONNX Runtime pipelines | conversion + quant + graph opt + tuning | Hardware-aware orchestration; outputs optimized ONNX. |
| **Hugging Face Optimum** | ✅ | Export + accelerate | ONNX graph opt + quant; backend integrations | Good glue across backends; not a single “one true optimizer”. |
| **TorchAO** | ⚠️ component | PyTorch-native low-bit + sparsity | quant + sparsity | Excellent building block; often used inside bigger stacks. |
| **LLM Compressor** | ✅-ish (LLMs) | vLLM serving | quant + pruning/sparsity | LLM-focused; pairs naturally with vLLM deployments. |

---

## 9) Handy links (docs + codebases)

### Pruna
- Docs home: https://docs.pruna.ai/  
- GitHub: https://github.com/PrunaAI/pruna  
- Tutorials index: https://docs.pruna.ai/en/stable/docs_pruna/tutorials/index.html  
- Algorithms Overview: https://docs.pruna.ai/en/stable/compression.html  
- SmashConfig guide: https://docs.pruna.ai/en/stable/docs_pruna/user_manual/configure.html  
- Evaluation: https://docs.pruna.ai/en/stable/reference/evaluation.html  
- Deploy integrations: https://docs.pruna.ai/en/stable/setup/integrations.html  
- PrunaAI (Hugging Face): https://huggingface.co/PrunaAI  

### Comparable “end-to-end” optimizers
- NVIDIA ModelOpt docs: https://nvidia.github.io/Model-Optimizer/  
- Intel Neural Compressor: https://intel.github.io/neural-compressor/  
- OpenVINO NNCF: https://github.com/openvinotoolkit/nncf  
- Microsoft Olive (ORT): https://onnxruntime.ai/docs/performance/olive.html  
- Hugging Face Optimum: https://huggingface.co/docs/optimum/en/index  
- LLM Compressor (vLLM): https://github.com/vllm-project/llm-compressor  
- TorchAO: https://github.com/pytorch/ao

---

## 10) Suggested practical steps
1) Pick **one representative workload** (inputs + constraints).  
2) Start with the **single highest-leverage lever** (LLMs → quantization; diffusion → caching).  
3) Add **one** more lever (compile or kernel) and re-evaluate.  
4) Save/push the best model + config so it’s reproducible.  
5) If you’re iterating a lot or need aggressive compression, consider **Pruna Pro’s OptimizationAgent** search + recovery path.
