# SmolVLA Quantization with Pruna

A rough example of optimizing **Vision–Language–Action (VLA)** policies using **[Pruna](https://www.pruna.ai/)** with SmolVLA from LeRobot.

I tried using **TorchAO quantization + torch.compile** for inference acceleration. Results are mixed, your run results may vary. I may extend this example to GPTQ later.

---

## Compatibility Notes (Pruna 0.2.10)
> At the time of writing, installing pruna==0.3.0 has some package dependency issues, so I opt for version 0.2.10.

> **Important:** Not all Pruna optimization methods are compatible with SmolVLA. After testing, here's what works:

| Method | Status | Notes |
|--------|--------|-------|
| **TorchAO int8wo** | ✅ Works | Weight-only INT8 quantization |
| **TorchAO int8dq** | ✅ Works | Dynamic INT8 quantization |
| **Half precision** | ✅ Works | FP16 conversion |
| **torch.compile** | ✅ Works | Graph compilation |
| TorchAO int4wo | ❌ Fails | Tensor shape error during quantization |
| HQQ | ❌ Fails | "Model is not compatible with hqq" |
| Structured pruning | ❌ Fails | "Model is not compatible with torch_structured" |
| Unstructured pruning | ❌ Fails | Not compatible |

---

## What this does

Three scripts form a complete optimization pipeline:

| Script | Purpose |
|--------|---------|
| `dump_smolvla_calib.py` | Generate calibration data from LeRobot datasets (for future pruning use) |
| `pruna_optimize_smolvla.py` | Apply Pruna optimizations (quantize + compile) |
| `bench_smolvla_pruna.py` | Benchmark baseline vs optimized variants |

---

## Optimization Stack

Pruna applies optimizations in order:

```
1. QUANTIZE         →    2. COMPILE
   ↓                         ↓
   Compress weights          Fuse operations
   to INT8                   for faster
   (TorchAO)                 inference
```

**Note:** Calibration data is not needed for TorchAO or half precision quantization.

---

## Requirements

- Python 3.11+
- LeRobot + SmolVLA: `pip install lerobot "lerobot[smolvla]"`
- Pruna: `pip install pruna==0.2.10`
- HuggingFace datasets: `pip install datasets`

```bash
# Full install
pip install lerobot "lerobot[smolvla]" pruna==0.2.10 datasets
```

---

## Quick Start

### 1. TorchAO INT8 (what worked for me)

**TorchAO INT8 weight-only** — seemed to give reasonable results:
```bash
python pruna_optimize_smolvla.py \
  --model-id lerobot/smolvla_base \
  --quantizer torchao \
  --torchao-quant-type int8wo \
  --out-dir ./out/torchao_int8wo
```

**TorchAO INT8 dynamic** — may preserve more quality:
```bash
python pruna_optimize_smolvla.py \
  --model-id lerobot/smolvla_base \
  --quantizer torchao \
  --torchao-quant-type int8dq \
  --out-dir ./out/torchao_int8dq
```

### 2. Half precision (FP16)

```bash
python pruna_optimize_smolvla.py \
  --model-id lerobot/smolvla_base \
  --quantizer half \
  --out-dir ./out/half_fp16
```

### 3. Compile only (no quantization)

```bash
python pruna_optimize_smolvla.py \
  --model-id lerobot/smolvla_base \
  --quantizer none \
  --out-dir ./out/compile_only
```

### 4. Benchmark all variants

```bash
python bench_smolvla_pruna.py \
  --baseline-id lerobot/smolvla_base \
  --smashed-dir ./out/torchao_int8wo ./out/torchao_int8dq ./out/half_fp16 \
  --export-csv results.csv
```

---

## Recommended Recipes

| Recipe | Command | Use Case |
|--------|---------|----------|
| **Best INT8 quality** | `--quantizer torchao --torchao-quant-type int8wo` | Production deployment |
| **Dynamic quant** | `--quantizer torchao --torchao-quant-type int8dq` | Quality-sensitive, activation quantization |
| **FP16 only** | `--quantizer half` | Simple precision reduction |
| **CPU deployment** | `--quantizer torchao --torchao-quant-type int8dq --no-compile` | CPU inference |
| **Ablation baseline** | `--quantizer none --no-compile` | Measure baseline without optimizations |

---

## CLI Reference

### `dump_smolvla_calib.py`

Generate calibration data from LeRobot datasets (for future pruning support).

```
--out-dir          Output directory for .npz files (required)
--dataset          HuggingFace dataset ID (default: lerobot/aloha_sim_insertion_human_image)
--split            Dataset split (default: train)
--n-samples        Number of calibration samples (default: 64)
--batch-size       Samples per .npz file (default: 8)
--image-size       Resize images to this size (default: 224)
--seed             Random seed for shuffling (default: 42)
--no-shuffle       Don't shuffle, take first N samples
```

**Note:** Use datasets with `observation.images.*` keys. The default `aloha_sim_insertion_human_image` works; `svla_so101_pickplace` does NOT have images.

### `pruna_optimize_smolvla.py`

Apply Pruna optimizations to SmolVLA.

**Model options:**
```
--model-id         HuggingFace model ID (default: lerobot/smolvla_base)
--out-dir          Output directory (default: ./out/smolvla_pruna_smashed)
--device           Device: cpu|cuda|mps (default: auto)
```

**Quantization:**
```
--quantizer        Backend: torchao|half|none (default: torchao)
--torchao-quant-type    int8wo|int8dq (default: int8dq on CPU, int8wo on CUDA)
--torchao-excluded-modules  none|norm|embedding|norm+embedding (default: norm+embedding)
```

**Compilation:**
```
--no-compile       Skip torch.compile
```

### `bench_smolvla_pruna.py`

Benchmark baseline vs optimized models.

```
--baseline-id      HuggingFace model ID for baseline
--smashed-dir      Path(s) to smashed model directories (required)
--device           Device: cpu|cuda|mps (default: auto)
--warmup-iters     Warmup iterations (default: 10, use 50+ for stable results)
--bench-iters      Benchmark iterations (default: 50, use 100+ for stable results)
--batch-size       Batch size (default: 1)
--task             Task instruction for tokenization (default: "pick up the object")
--export-csv       Export results to CSV file
--skip-baseline    Skip baseline benchmark
```

---

## Example Output

*Note: These numbers are illustrative. Actual results depend on your hardware, model version, and workload.*

### Optimization output

```
[pruna-smolvla] Device: cuda
[pruna-smolvla] Output: ./out/torchao_int8wo
[pruna-smolvla] Loading model: lerobot/smolvla_base
[pruna-smolvla] Baseline parameters: 100,000,000
[pruna-smolvla] Configuring quantizer: torchao
[pruna-smolvla]   torchao_quant_type: int8wo
[pruna-smolvla]   torchao_excluded_modules: norm+embedding
[pruna-smolvla] Configuring compiler: torch_compile

[pruna-smolvla] Running Pruna smash...
[pruna-smolvla] Optimization stack: quantize(torchao, int8wo) → compile(torch_compile)

============================================================
[pruna-smolvla] Optimization complete!
============================================================
  Model ID:       lerobot/smolvla_base
  Device:         cuda
  Quantizer:      torchao (int8wo)
  Compilation:    enabled
  Baseline params: 100,000,000
  Output saved:   True
  Output dir:     ./out/torchao_int8wo
============================================================
```

### Benchmark output

```
==========================================================================================
BENCHMARK RESULTS
==========================================================================================
Model                              Params    Latency (ms)   Memory (MB)    Speedup
                                              mean ± std          peak      vs base
------------------------------------------------------------------------------------------
baseline                      100,000,000      25.50 ± 1.50        2500.0       1.00x
torchao_int8wo                100,000,000      18.30 ± 1.20        1500.0       1.39x
torchao_int8dq                100,000,000      19.80 ± 1.35        1550.0       1.29x
half_fp16                     100,000,000      20.10 ± 1.40        1250.0       1.27x
==========================================================================================

Best speedup: 1.39x
Best memory savings: 50.0%
```

*Take these numbers with a grain of salt — I saw high variance in my runs, and your setup may differ.*

---

## Folder Layout

```
examples/pruna-smolvla/
├── README.md                    # This file
├── dump_smolvla_calib.py        # Calibration data generator
├── pruna_optimize_smolvla.py    # Main optimization script
├── bench_smolvla_pruna.py       # Benchmarking script
├── calib_data/                  # Generated calibration data (optional)
│   ├── calib_00000.npz
│   └── ...
└── out/                         # Optimized models
    ├── torchao_int8wo/
    │   └── smash_config.json
    └── torchao_int8dq/
        └── smash_config.json
```

---

## Validating Results (Don't Skip This!)

Optimized models must be validated on your actual robot task:

### 1. Memory & latency (automated)
```bash
python bench_smolvla_pruna.py --smashed-dir ./out/your_model
```

### 2. Policy quality (manual)
- Run the optimized policy on your robot (real or sim)
- Compare success rate vs baseline on a fixed eval set
- Check for behavior drift (does the robot still move smoothly?)

**Warning:** A model with great latency but broken policy behavior is useless!

---

## Caveats & Gotchas

### Quantization
- **INT8 is reliable.** Both `int8wo` and `int8dq` work well with SmolVLA.
- **INT4 doesn't work.** TorchAO INT4 fails with tensor shape errors on SmolVLA.
- **HQQ doesn't work.** Pruna 0.2.10 reports "Model is not compatible with hqq" for SmolVLA.
- **Exclude norms/embeddings.** `--torchao-excluded-modules norm+embedding` is the safe default.

### Pruning
- **Not currently supported.** Pruna 0.2.10 reports "Model is not compatible with torch_structured" for SmolVLA.
- Future Pruna versions may add support.

### Compilation
- **Warmup is critical.** `torch.compile` models are slow on first inference. Always benchmark after warmup.
- **Keep a no-compile baseline.** Use `--no-compile` to isolate compilation effects from quantization.

---

## References

- [Pruna docs](https://docs.pruna.ai/)
- [Pruna Smash API](https://docs.pruna.ai/docs/smash)
- [TorchAO](https://github.com/pytorch/ao)
- [LeRobot](https://github.com/huggingface/lerobot)
- [SmolVLA model](https://huggingface.co/lerobot/smolvla_base)
- [Model optimization notes](../../notes/Pruna.md)

---

## License

This example is provided as-is. Check upstream licenses for:
- LeRobot + SmolVLA
- Pruna
- Model checkpoints from Hugging Face
