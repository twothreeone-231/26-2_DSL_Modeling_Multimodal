"""
Run the frozen AST backbone once over the mixed dataset and cache the pooled
embeddings to disk.

This machine has no GPU, so a full AST forward costs ~2.6 s/sample on CPU.
Fine-tuning the backbone is not feasible; instead we pay that cost exactly once
per sample here and then train the classifier head on the cached vectors
(train_head.py), which takes seconds.

Resumable: embeddings are appended to a shard file per split, and a rerun skips
rows that are already cached.

    python extract_embeddings.py --split train --limit 6000
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import MixedASTDataset
from model import MODEL_NAME

from transformers import ASTModel


# Dynamically derived (matches train.py/evaluation.py/dataset.py) instead of a
# hardcoded /data1/... string, which points at an unrelated local directory on
# compute nodes (the shared storage is only at /mnt/data1/... there).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_DIR = PROJECT_ROOT / "yusung/data_processed_v2/metadata"
CACHE_DIR = Path(__file__).resolve().parent / "embeddings"

LABEL_COLS = ["VF", "MF", "FF"]


def pooled_embedding(ast, input_values):
    """CLS + distillation token average, matching DeepVoiceAST.forward."""
    outputs = ast(input_values=input_values)
    hidden = outputs.last_hidden_state
    return (hidden[:, 0] + hidden[:, 1]) / 2


def load_cache(cache_path):
    if not cache_path.exists():
        return None
    return np.load(cache_path, allow_pickle=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True,
                        choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of samples (stratified by combo). "
                             "Use this to get a working model without the full "
                             "~20 h CPU pass over every split.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = METADATA_DIR / f"mixed_chunks_{args.split}.csv"
    df = pd.read_csv(csv_path)

    if args.limit is not None and args.limit < len(df):
        # Keep the 4 combos balanced, exactly as the full split is.
        per_combo = args.limit // df["combo"].nunique()
        df = (
            df.groupby("combo", group_keys=False)
            .apply(lambda g: g.sample(per_combo, random_state=args.seed))
            .sort_index()
            .reset_index(drop=True)
        )

    tag = f"{args.split}" if args.limit is None else f"{args.split}_n{len(df)}"
    cache_path = CACHE_DIR / f"{tag}.npz"

    cached = load_cache(cache_path)
    done_ids = set(cached["ids"].tolist()) if cached is not None else set()

    todo = df[~df["mixed_chunk_id"].isin(done_ids)].reset_index(drop=True)

    print(f"split={args.split} total={len(df):,} "
          f"cached={len(done_ids):,} todo={len(todo):,}")

    if len(todo) == 0:
        print("nothing to do")
        return

    # MixedASTDataset reads from a CSV path, so hand it the subset we still owe.
    todo_csv = CACHE_DIR / f".todo_{tag}.csv"
    todo.to_csv(todo_csv, index=False)

    dataset = MixedASTDataset(todo_csv)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    ast = ASTModel.from_pretrained(MODEL_NAME)
    ast.eval()
    for param in ast.parameters():
        param.requires_grad = False

    embeddings = [cached["embeddings"]] if cached is not None else []
    ids = [cached["ids"]] if cached is not None else []
    labels = [cached["labels"]] if cached is not None else []
    combos = [cached["combos"]] if cached is not None else []

    new_emb, new_ids, new_labels, new_combos = [], [], [], []

    start = time.time()
    done = 0

    def flush():
        """Write everything gathered so far, so a kill never loses the pass."""
        if not new_emb:
            return
        np.savez(
            cache_path,
            embeddings=np.concatenate(embeddings + [np.concatenate(new_emb)]),
            ids=np.concatenate(ids + [np.array(new_ids)]),
            labels=np.concatenate(labels + [np.concatenate(new_labels)]),
            combos=np.concatenate(combos + [np.array(new_combos)]),
        )

    with torch.no_grad():
        for step, batch in enumerate(loader):
            emb = pooled_embedding(ast, batch["input_values"])

            new_emb.append(emb.numpy().astype(np.float32))
            new_labels.append(batch["labels"].numpy().astype(np.float32))

            lo = step * args.batch_size
            chunk = todo.iloc[lo:lo + emb.shape[0]]
            new_ids.extend(chunk["mixed_chunk_id"].tolist())
            new_combos.extend(chunk["combo"].tolist())

            done += emb.shape[0]

            if step % 25 == 0 or done == len(todo):
                elapsed = time.time() - start
                rate = done / elapsed
                eta = (len(todo) - done) / rate / 60
                print(f"[{done:,}/{len(todo):,}] "
                      f"{rate:.2f} samp/s  ETA {eta:.1f} min", flush=True)

            if step % 100 == 0 and step > 0:
                flush()

    flush()
    todo_csv.unlink(missing_ok=True)
    print(f"saved -> {cache_path}")


if __name__ == "__main__":
    main()
