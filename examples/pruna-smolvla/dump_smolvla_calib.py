"""
Dump calibration data for SmolVLA from LeRobot datasets.

Streams a LeRobot dataset, grabs observations in SmolVLA format, and saves
them as .npz files. Originally intended for Pruna pruning, but pruning didn't
work with SmolVLA (kept anyway in case future versions support it).

Usage:
  python dump_smolvla_calib.py --out-dir ./calib_data --n-samples 64

Output .npz files contain:
  - images: camera views as CHW float16
  - state: proprioceptive state as float16
  - task: task instruction string
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def pil_to_chw_float16(pil_img: Image.Image, image_size: int = 224) -> np.ndarray:
    """PIL image -> CHW float16 array, normalized to [0,1]."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    if pil_img.size != (image_size, image_size):
        pil_img = pil_img.resize((image_size, image_size), resample=Image.BILINEAR)
    arr = np.asarray(pil_img, dtype=np.uint8)
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = arr.astype(np.float16) / 255.0
    return arr


def extract_image_keys(example: dict) -> list[str]:
    """Find all observation.images.* keys in a dataset sample."""
    image_keys = [k for k in example.keys() if k.startswith("observation.images.")]
    if not image_keys:
        print("[smolvla-calib] Warning: No image keys found in dataset.")
        print("[smolvla-calib] Available keys:", list(example.keys()))
        print("[smolvla-calib] Consider using a dataset with images, e.g.:")
        print("[smolvla-calib]   --dataset lerobot/aloha_sim_insertion_human_image")
    return image_keys


def process_example(
    example: dict,
    image_keys: list[str],
    image_size: int = 224,
) -> dict[str, Any]:
    """Convert one dataset sample to SmolVLA format (images, state, task)."""
    # Process images
    images = {}
    for key in image_keys:
        img = example[key]
        if isinstance(img, Image.Image):
            camera_name = key.replace("observation.images.", "")
            images[camera_name] = pil_to_chw_float16(img, image_size)
    
    # Process state
    state = np.asarray(example.get("observation.state", []), dtype=np.float16)
    
    # Process task instruction
    task = ""
    if "task" in example:
        task = str(example["task"])
    elif "language_instruction" in example:
        task = str(example["language_instruction"])
    elif "task_index" in example:
        task_idx = int(np.asarray(example["task_index"]).reshape(-1)[0])
        task = f"task {task_idx}"
    
    return {
        "images": images,
        "state": state,
        "task": task,
    }


def save_batch(
    samples: list[dict],
    out_path: Path,
    batch_idx: int,
) -> None:
    """Write a batch of samples to calib_XXXXX.npz."""
    camera_names = list(samples[0]["images"].keys())
    
    # Stack images by camera
    images_dict = {}
    for cam in camera_names:
        images_dict[f"images.{cam}"] = np.stack([s["images"][cam] for s in samples], axis=0)
    
    states = np.stack([s["state"] for s in samples], axis=0)
    tasks = [s["task"] for s in samples]
    
    np.savez_compressed(
        out_path / f"calib_{batch_idx:05d}.npz",
        **images_dict,
        state=states,
        tasks=json.dumps(tasks),
    )
    
    print(
        f"[smolvla-calib] wrote calib_{batch_idx:05d}.npz  "
        f"cameras={camera_names} state={tuple(states.shape)} n_tasks={len(tasks)}",
        flush=True,
    )


class SmolVLACalibDataset(Dataset):
    """
    Loads our .npz calibration files as a PyTorch Dataset.
    Meant for Pruna's cfg.add_data(), though pruning didn't work for SmolVLA.
    """
    
    def __init__(self, calib_dir: str | Path, device: str = "cpu"):
        self.calib_dir = Path(calib_dir)
        self.device = device
        self.npz_files = sorted(self.calib_dir.glob("calib_*.npz"))
        
        if not self.npz_files:
            raise ValueError(f"No calibration files found in {calib_dir}")
        
        # Load all samples into memory (calibration sets are typically small)
        self.samples = []
        for npz_path in self.npz_files:
            data = np.load(npz_path, allow_pickle=True)
            tasks = json.loads(str(data["tasks"]))
            
            # Find image keys
            image_keys = [k for k in data.files if k.startswith("images.")]
            batch_size = len(tasks)
            
            for i in range(batch_size):
                sample = {
                    "images": {},
                    "state": data["state"][i],
                    "task": tasks[i],
                }
                for img_key in image_keys:
                    cam_name = img_key.replace("images.", "")
                    sample["images"][cam_name] = data[img_key][i]
                self.samples.append(sample)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        
        # Convert to tensors in LeRobot observation format
        observation = {}
        
        # Images: observation.images.{camera_name}
        for cam_name, img_arr in sample["images"].items():
            key = f"observation.images.{cam_name}"
            observation[key] = torch.from_numpy(img_arr.astype(np.float32))
        
        # State: observation.state
        observation["observation.state"] = torch.from_numpy(
            sample["state"].astype(np.float32)
        )
        
        return {
            "observation": observation,
            "task": sample["task"],
        }


def smolvla_collate_fn(batch: list[dict]) -> dict:
    """Stack observations into batched tensors for SmolVLA."""
    # Find all observation keys from first sample
    obs_keys = list(batch[0]["observation"].keys())
    
    # Stack observations
    observation = {}
    for key in obs_keys:
        tensors = [sample["observation"][key] for sample in batch]
        observation[key] = torch.stack(tensors, dim=0)
    
    # Collect tasks
    tasks = [sample["task"] for sample in batch]
    
    return {
        "observation": observation,
        "task": tasks,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate calibration data for SmolVLA from LeRobot datasets"
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for calibration .npz files",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default="lerobot/svla_so101_pickplace",
        help="HuggingFace dataset ID (default: lerobot/svla_so101_pickplace)",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    ap.add_argument(
        "--n-samples",
        type=int,
        default=64,
        help="Total number of calibration samples (default: 64)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Samples per .npz file (default: 8)",
    )
    ap.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Resize images to this size (default: 224)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    ap.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle, take first N samples (faster for testing)",
    )
    args = ap.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset with streaming to avoid downloading everything
    from datasets import load_dataset
    
    print(f"[smolvla-calib] Loading dataset: {args.dataset} (split={args.split})")
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    
    if args.no_shuffle:
        it = ds.take(args.n_samples)
    else:
        it = ds.shuffle(buffer_size=2000, seed=args.seed).take(args.n_samples)
    
    # Process samples
    buf = []
    batch_idx = 0
    image_keys = None
    
    for ex in it:
        # Detect image keys from first example
        if image_keys is None:
            image_keys = extract_image_keys(ex)
            print(f"[smolvla-calib] Detected image keys: {image_keys}")
        
        sample = process_example(ex, image_keys, args.image_size)
        buf.append(sample)
        
        if len(buf) >= args.batch_size:
            save_batch(buf, out_dir, batch_idx)
            batch_idx += 1
            buf = []
    
    if buf:
        save_batch(buf, out_dir, batch_idx)
        batch_idx += 1
    
    print(f"\n[smolvla-calib] Done! Generated {batch_idx} batch files in {out_dir}")
    print(f"[smolvla-calib] Total samples: {args.n_samples}")
    print(f"\nTo use with pruna_optimize_smolvla.py:")
    print(f"  python pruna_optimize_smolvla.py --prune --calib-dir {out_dir}")


if __name__ == "__main__":
    main()
