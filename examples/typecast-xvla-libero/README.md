# X-VLA (LIBERO) Typecast Benchmark

This folder contains a small  example harness that benchmarks **`typecast-xvla-libero`** on the **LIBERO simulation benchmark** while trying a few inference optimizations (AMP bf16 or fp16, TF32, Flash SDP). It produces per run timing, an inferred success rate (if present in LeRobot’s eval artifacts), and a separate **policy only latency** metric.

## Prerequisites

- NVIDIA GPU + CUDA for meaningful speed comparisons
- Python 3.10+ (or the version required by your LeRobot setup)

## Installation

1. Clone and install LeRobot with LIBERO and X-VLA extras:

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# Install LeRobot in editable mode + required extras
pip install -e ".[libero]"
pip install -e ".[xvla]"
````

2. LIBERO assets and environment setup

Follow LeRobot’s LIBERO setup (datasets, assets) as you normally do for `lerobot-eval`. If you already can run:

```bash
lerobot-eval --env.type=libero --env.task=libero_spatial
```

then you are good.

3. Headless rendering

LIBERO uses MuJoCo. For headless GPU rendering:

```bash
export MUJOCO_GL=egl
```

If EGL is not available on your machine, you may need OSMesa or a display, depending on your setup.

## Files

* `xvla_libero_bench.py`: the benchmark runner
* Output directory (default): `./bench_xvla_libero/`

  * `results.jsonl`: one JSON record per run
  * `results.csv`: compact summary table
  * `<run_name>/stdout.log`, `<run_name>/stderr.log`
  * `<run_name>/policy_latency.json` (if the patch loads), contains mean/p50/p90 latency of `policy.select_action`

## Run

From the directory where `xvla_libero_bench.py` lives:

```bash
export MUJOCO_GL=egl

python xvla_libero_bench.py \
  --output_root ./bench_xvla_libero \
  --policy_path lerobot/xvla-libero \
  --tasks libero_spatial,libero_goal \
  --control_mode absolute \
  --episode_length 800 \
  --batch_size 1 \
  --n_episodes 5 \
  --seed 142 \
  --max_parallel_tasks 1
```

## Interpreting results

* **Wall time / episodes per second**: end to end eval speed, often dominated by simulation and rendering.
* **`policy_latency.json`**: the most direct “model inference speed” number. Use this to compare optimizations fairly.
* **Success rate**: extracted from LeRobot’s `eval_info.json` (or similar). If success is missing, it will be blank.

## Common troubleshooting

### bf16 fails when using `--policy.dtype=bfloat16`

This harness does **not** force cast model weights via CLI. Instead it uses **AMP autocast** via a `sitecustomize.py` patch, which is typically more stable (LayerNorm and similar ops can remain fp32 internally).

### Success rate is blank

Check the output directory for `eval_info.json`. If it contains no key related to success, LeRobot did not save it for that run, and the harness cannot infer it.

### Speedup is small

That can be expected if the rollout is simulation bound. Use `policy_latency.json` to see if the model got faster even if wall time did not change much.

## Reproducibility tips

* Keep `--seed` fixed
* Keep `--max_parallel_tasks=1` for more stable timing
* Run each configuration multiple times and average `policy_latency.json`
