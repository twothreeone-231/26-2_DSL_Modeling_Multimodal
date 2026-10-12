"""Evaluate a trained DeepVoiceAST checkpoint on validation and test splits."""

import argparse
import json
from pathlib import Path

import torch

from model import DeepVoiceAST
from train import (
    LABEL_NAMES,
    TEST_CSV,
    TRAIN_CSV,
    VAL_CSV,
    build_loader,
    compute_pos_weight,
    masked_bce_loss,
    evaluate as evaluate_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "ast_best.pt"


def evaluate_checkpoint(checkpoint_path, split, batch_size, num_workers,
                        device, amp_enabled):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    saved_args = checkpoint.get("args", {})
    dropout = saved_args.get("dropout", 0.2)
    seed = saved_args.get("seed", 42)
    train_limit = saved_args.get("limit")

    model = DeepVoiceAST(
        dropout=dropout,
        freeze_backbone=False,
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    pos_weight = compute_pos_weight(TRAIN_CSV, train_limit, seed, device)

    def criterion(logits, labels, mask):
        return masked_bce_loss(logits, labels, mask, pos_weight)

    amp_dtype = torch.bfloat16 if amp_enabled and device.type == "cuda" else None

    split_config = {
        "val": (VAL_CSV, saved_args.get("val_limit")),
        "test": (TEST_CSV, None),
    }
    csv_path, limit = split_config[split]
    loader, groups = build_loader(
        csv_path,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        limit=limit,
        seed=seed,
    )

    metrics = evaluate_split(
        model,
        loader,
        groups,
        criterion,
        device,
        amp_dtype,
    )
    return {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "samples": len(groups),
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Path to ast_best.pt produced by train.py.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test", "both"],
        default="both",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=None)
    parser.add_argument(
        "--report",
        default=None,
        help="Output JSON path. Defaults beside the checkpoint.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    saved_amp = checkpoint.get("args", {}).get("amp", False)
    amp_enabled = saved_amp if args.amp is None else args.amp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  amp: {amp_enabled and device.type == 'cuda'}")

    splits = ["val", "test"] if args.split == "both" else [args.split]
    results = {}
    for split in splits:
        result = evaluate_checkpoint(
            checkpoint_path,
            split,
            args.batch_size,
            args.num_workers,
            device,
            amp_enabled,
        )
        results[split] = result
        print(json.dumps(result, indent=2))

    report_path = Path(args.report) if args.report else (
        checkpoint_path.parent / "evaluation_report.json"
    )
    report_path.write_text(json.dumps(results, indent=2))
    print(f"saved evaluation report -> {report_path}")


if __name__ == "__main__":
    main()
