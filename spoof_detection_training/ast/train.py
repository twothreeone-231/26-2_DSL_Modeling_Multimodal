"""
Train DeepVoiceAST on the v3 mixed dataset.

NOTE: in the current pipeline split (see yubeen/README.md), AST only owns
presence (VP/MP, in presence.py) -- VF/MF/FF fake detection moved to AASIST
(yubeen/aasist/train.py), which presence.py now loads for the fake side of
its 5-head combine. This script and DeepVoiceAST are kept as a baseline /
comparison point, not the production fake-detector.

Each 10-second clip is one of 9 label groups (voice_only, voice_fake_only,
music_only, music_fake_only, voice_music, voice_music_fake, voice_fake_music,
voice_fake_music_fake, noise_only) -- unlike v2, voice and music are not
always both present. The model answers three questions independently with
one sigmoid per axis:

    VF : the voice is fake       (label -1, mask 0, when there is no voice)
    MF : the music is fake       (label -1, mask 0, when there is no music)
    FF : the file contains anything fake among the components present

Loss and metrics only look at (label, mask) pairs where mask == 1, matching
the competition's own rule that Voice EER / Music EER are computed only on
samples where that component is present.

    python train.py --epochs 10 --batch-size 16
    python train.py --unfreeze-epoch 2 --backbone-lr 1e-5   # fine-tune backbone
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "yubeen"))

from common import (  # noqa: E402
    balanced_subset,
    compute_masked_pos_weight,
    evaluate_masked_multilabel,
    group_from_mixed_id,
    masked_bce_loss,
)
from dataset import FAKE_LABELS as LABEL_NAMES
from dataset import MixedASTDataset
from model import DeepVoiceAST


METADATA_DIR = PROJECT_ROOT / "yusung/data_mixed_v5/metadata"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

TRAIN_CSV = METADATA_DIR / "mixed_train.csv"
VAL_CSV = METADATA_DIR / "mixed_val.csv"
TEST_CSV = METADATA_DIR / "mixed_test.csv"

# Subsets for --limit runs are written here, not into the shared dataset dir.
SUBSET_DIR = Path(__file__).resolve().parent / "subsets"


def build_loader(csv_path, batch_size, num_workers, shuffle, limit, seed):
    """Build a loader over a split, optionally on a group-balanced subset."""
    df = pd.read_csv(csv_path)

    if limit is not None:
        df = balanced_subset(df, limit, seed)
        SUBSET_DIR.mkdir(parents=True, exist_ok=True)
        subset_csv = SUBSET_DIR / f"{Path(csv_path).stem}_{limit}.csv"
        df.to_csv(subset_csv, index=False)
        csv_path = subset_csv

    groups = df["mixed_id"].map(group_from_mixed_id).to_numpy()

    dataset = MixedASTDataset(csv_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader, groups


def compute_pos_weight(csv_path, limit, seed, device):
    """Per-axis class weight, counting only rows where that axis has a label."""
    df = pd.read_csv(csv_path)
    if limit is not None:
        df = balanced_subset(df, limit, seed)
    return compute_masked_pos_weight(df, LABEL_NAMES, device)


# Same masked multi-label eval AASIST uses for VF/MF/FF -- see
# yubeen/common.py for the ads_like / per_group_ff_acc definitions.
evaluate = evaluate_masked_multilabel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--unfreeze-epoch", type=int, default=None,
                        help="Epoch (1-indexed) at which to unfreeze the AST "
                             "backbone. Omit to keep it frozen throughout.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap samples per split, kept balanced across the "
                             "9 label groups. Useful for a quick sanity run.")
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--amp", action="store_true",
                        help="Mixed precision (bf16 on CUDA).")
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(DEFAULT_CHECKPOINT_DIR))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (args.amp and device.type == "cuda") else None
    print(f"device: {device}  amp: {amp_dtype}")

    train_loader, train_groups = build_loader(
        TRAIN_CSV, args.batch_size, args.num_workers,
        shuffle=True, limit=args.limit, seed=args.seed,
    )
    val_loader, val_groups = build_loader(
        VAL_CSV, args.batch_size, args.num_workers,
        shuffle=False, limit=args.val_limit, seed=args.seed,
    )

    model = DeepVoiceAST(
        dropout=args.dropout,
        freeze_backbone=True,
    ).to(device)

    pos_weight = compute_pos_weight(TRAIN_CSV, args.limit, args.seed, device)
    print("pos_weight:", dict(zip(LABEL_NAMES, pos_weight.tolist())))

    def criterion(logits, labels, mask):
        return masked_bce_loss(logits, labels, mask, pos_weight)

    # Two groups from the start: the backbone group sits at lr 0 impact while
    # frozen, and starts training the moment we unfreeze.
    optimizer = torch.optim.AdamW(
        [
            {"params": model.classifier.parameters(), "lr": args.head_lr},
            {"params": model.ast.parameters(), "lr": args.backbone_lr},
        ],
        weight_decay=args.weight_decay,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ast_best.pt"

    best_ads_like = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        if args.unfreeze_epoch is not None and epoch == args.unfreeze_epoch:
            print(f"--- unfreezing AST backbone at epoch {epoch} ---")
            model.unfreeze_backbone()

        model.train()
        running = 0.0
        seen = 0

        for step, batch in enumerate(train_loader):
            inputs = batch["input_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                loss = criterion(model(inputs), labels, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item() * inputs.shape[0]
            seen += inputs.shape[0]

            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)}  "
                      f"loss {running / seen:.4f}", flush=True)

        metrics = evaluate(model, val_loader, val_groups,
                           criterion, device, amp_dtype)
        eer = metrics["per_label_eer"]
        print(f"[epoch {epoch}] train_loss {running / seen:.4f}  "
              f"val_loss {metrics['loss']:.4f}  "
              f"EER(FF/VF/MF) {eer['FF']:.3f}/{eer['VF']:.3f}/{eer['MF']:.3f}  "
              f"ads_like {metrics['ads_like']:.4f}", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": running / seen,
            **metrics,
        })

        if metrics["ads_like"] > best_ads_like:
            best_ads_like = metrics["ads_like"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "args": vars(args),
                    "val_metrics": metrics,
                },
                ckpt_path,
            )
            print(f"  saved best -> {ckpt_path}")

    print(f"\nbest val ads_like: {best_ads_like:.4f}")
    result = {"best_val_ads_like": best_ads_like, "history": history}

    if args.eval_test:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        test_loader, test_groups = build_loader(
            TEST_CSV, args.batch_size, args.num_workers,
            shuffle=False, limit=None, seed=args.seed,
        )
        test_metrics = evaluate(model, test_loader, test_groups,
                                criterion, device, amp_dtype)
        result["test"] = test_metrics
        print("\n=== test ===")
        print(json.dumps(test_metrics, indent=2))

    (out_dir / "training_report.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
