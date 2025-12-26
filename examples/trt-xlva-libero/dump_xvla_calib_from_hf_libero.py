#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast


def pil_to_chw_float16(pil_img: Image.Image, image_size: int) -> np.ndarray:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    if pil_img.size != (image_size, image_size):
        pil_img = pil_img.resize((image_size, image_size), resample=Image.BILINEAR)
    arr = np.asarray(pil_img, dtype=np.uint8)              # HWC uint8
    arr = arr.transpose(2, 0, 1)                           # CHW
    arr = (arr.astype(np.float16) / np.float16(255.0))     # CHW float16 in [0, 1]
    return arr


def load_xvla_tokenizer(repo_id: str) -> PreTrainedTokenizerFast:
    # X-VLA repos include tokenizer.json
    tok_json = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
    tok = PreTrainedTokenizerFast(tokenizer_file=tok_json)

    # Ensure padding works
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--dataset", type=str, default="HuggingFaceVLA/libero")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--tokenizer_repo", type=str, default="2toINF/X-VLA-Libero")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--n_batches", type=int, default=16)
    ap.add_argument("--max_seq_len", type=int, default=48)
    ap.add_argument("--image_size", type=int, default=192)
    ap.add_argument("--domain_id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=142)
    ap.add_argument("--take_first", action="store_true", help="No shuffle, just take first N samples (fastest)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = load_xvla_tokenizer(args.tokenizer_repo)

    total = args.batch_size * args.n_batches
    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    if args.take_first:
        it = ds.take(total)
    else:
        it = ds.shuffle(buffer_size=2000, seed=args.seed).take(total)

    buf = []
    batch_idx = 0

    for ex in it:
        buf.append(ex)
        if len(buf) < args.batch_size:
            continue

        input_ids_list = []
        image_list = []
        image_mask_list = []
        proprio_list = []

        for e in buf:
            img1 = e["observation.images.image"]
            img2 = e["observation.images.image2"]
            state = np.asarray(e["observation.state"], dtype=np.float16)

            task_index = int(np.asarray(e["task_index"]).reshape(-1)[0])
            prompt = f"libero task {task_index}"

            enc = tok(
                prompt,
                padding="max_length",
                truncation=True,
                max_length=args.max_seq_len,
                return_tensors="np",
            )

            v1 = pil_to_chw_float16(img1, args.image_size)
            v2 = pil_to_chw_float16(img2, args.image_size)
            img = np.stack([v1, v2], axis=0)  # [V, C, H, W]

            input_ids_list.append(enc["input_ids"][0].astype(np.int32))
            image_list.append(img)
            image_mask_list.append(np.ones((2,), dtype=np.int32))
            proprio_list.append(state)

        input_ids = np.stack(input_ids_list, axis=0)               # [B, L]
        image_input = np.stack(image_list, axis=0)                 # [B, V, C, H, W]
        image_mask = np.stack(image_mask_list, axis=0)             # [B, V]
        proprio = np.stack(proprio_list, axis=0)                   # [B, D]
        domain_id = np.full((args.batch_size,), args.domain_id, dtype=np.int32)

        np.savez_compressed(
            out_dir / f"calib_{batch_idx:05d}.npz",
            input_ids=input_ids,
            image_input=image_input,
            image_mask=image_mask,
            proprio=proprio,
            domain_id=domain_id,
        )

        print(
            f"[xvla-calib] wrote calib_{batch_idx:05d}.npz  "
            f"input_ids={tuple(input_ids.shape)} image_input={tuple(image_input.shape)}",
            flush=True,
        )

        batch_idx += 1
        buf = []


if __name__ == "__main__":
    main()
