"""
Train the DeepVoiceAST classifier head on cached AST embeddings.

Targets are multi-label, one sigmoid per axis:

    VF : the voice in this clip is fake
    MF : the music in this clip is fake
    FF : the file contains anything fake  (== VF or MF)

So a single clip answers both questions the project cares about --
"진짜 음성 / 가짜 음성" and "진짜 음악 / 가짜 음악" -- and the 4 combos of the
mixed dataset fall out as the (VF, MF) pairs.

    python train_head.py --train-cache embeddings/train_n6000.npz \
                         --val-cache   embeddings/val_n1000.npz \
                         --test-cache  embeddings/test_n1000.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


LABEL_NAMES = ["VF", "MF", "FF"]
COMBO_ORDER = [
    "real_voice_real_music",
    "real_voice_fake_music",
    "fake_voice_real_music",
    "fake_voice_fake_music",
]


def load_cache(path):
    data = np.load(path, allow_pickle=True)
    x = torch.from_numpy(data["embeddings"].astype(np.float32))
    y = torch.from_numpy(data["labels"].astype(np.float32))
    return x, y, data["combos"]


def build_head(in_dim, hidden, dropout):
    if hidden == 0:
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, 3))
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Dropout(dropout),
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 3),
    )


@torch.no_grad()
def evaluate(head, x, y, combos):
    head.eval()
    logits = head(x)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()

    per_label = {
        name: (preds[:, i] == y[:, i]).float().mean().item()
        for i, name in enumerate(LABEL_NAMES)
    }

    # Exact match on the (VF, MF) pair -- i.e. the 4-way combo is fully right.
    combo_exact = (preds[:, :2] == y[:, :2]).all(dim=1).float()

    per_combo = {}
    for name in COMBO_ORDER:
        mask = torch.from_numpy(combos == name)
        if mask.any():
            per_combo[name] = combo_exact[mask].mean().item()

    return {
        "per_label_acc": per_label,
        "combo_exact_acc": combo_exact.mean().item(),
        "per_combo_acc": per_combo,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--test-cache", default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256,
                        help="0 for a plain linear probe")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="checkpoints")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x_train, y_train, c_train = load_cache(args.train_cache)
    x_val, y_val, c_val = load_cache(args.val_cache)

    print(f"train {tuple(x_train.shape)}  val {tuple(x_val.shape)}")

    # Standardize on train statistics; AST pooled features are not centered.
    mean = x_train.mean(0, keepdim=True)
    std = x_train.std(0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    head = build_head(x_train.shape[1], args.hidden, args.dropout)

    # FF is 1 far more often than 0 (3 of the 4 combos), so weight it back.
    pos = y_train.sum(0)
    neg = y_train.shape[0] - pos
    pos_weight = (neg / pos.clamp_min(1)).clamp(0.1, 10.0)
    print("pos_weight:", dict(zip(LABEL_NAMES, pos_weight.tolist())))

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "head_best.pt"

    best = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        head.train()
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(head(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.shape[0]
        scheduler.step()

        train_loss = total / len(x_train)
        metrics = evaluate(head, x_val, y_val, c_val)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})

        if epoch % 5 == 0 or epoch == 1:
            acc = metrics["per_label_acc"]
            print(f"epoch {epoch:3d}  loss {train_loss:.4f}  "
                  f"VF {acc['VF']:.3f}  MF {acc['MF']:.3f}  "
                  f"combo {metrics['combo_exact_acc']:.3f}", flush=True)

        if metrics["combo_exact_acc"] > best:
            best = metrics["combo_exact_acc"]
            torch.save(
                {
                    "head": head.state_dict(),
                    "mean": mean,
                    "std": std,
                    "args": vars(args),
                    "val_metrics": metrics,
                    "epoch": epoch,
                },
                ckpt_path,
            )

    print(f"\nbest val combo accuracy: {best:.4f} -> {ckpt_path}")

    result = {"best_val_combo_acc": best, "history": history}

    if args.test_cache:
        state = torch.load(ckpt_path, weights_only=False)
        head.load_state_dict(state["head"])
        x_test, y_test, c_test = load_cache(args.test_cache)
        x_test = (x_test - mean) / std
        test_metrics = evaluate(head, x_test, y_test, c_test)
        result["test"] = test_metrics
        print("\n=== test ===")
        print(json.dumps(test_metrics, indent=2))

    (out_dir / "training_report.json").write_text(
        json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
