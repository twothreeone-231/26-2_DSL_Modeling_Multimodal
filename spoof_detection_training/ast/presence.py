"""Presence and combined 5-head inference utilities for the DeepVoice pipeline.

This script handles two tasks:

1) Train/evaluate a presence head for VP / MP labels (AST backbone).
2) Combine AASIST's 3 fake-detection heads (VF, MF, FF) with AST's presence
   heads (VP, MP), producing a total of 5 outputs for an actual audio file.

Fake detection (VF/MF/FF) is answered by AASIST (yubeen/aasist), not AST --
see yubeen/README.md for the pipeline split. Only presence stays on the AST
side; the combine step below loads an AASIST checkpoint for the fake half.

Typical usage:
    python yubeen/ast/presence.py --epochs 10 --batch-size 16
    python yubeen/ast/presence.py --audio /path/to/audio.wav \
        --fake-checkpoint ../aasist/checkpoints/aasist_best.pt \
        --presence-checkpoint checkpoints/presence_best.pt
    python yubeen/ast/presence.py --audio-dir data/test_audio --fake-checkpoint ... --presence-checkpoint ...
"""

import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import ASTFeatureExtractor, ASTModel

from train import group_from_mixed_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_DIR = PROJECT_ROOT / "yusung/data_mixed_v5/metadata"
AUDIO_DIR = PROJECT_ROOT / "yusung/data_mixed_v5/audio"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

TRAIN_CSV = METADATA_DIR / "mixed_train.csv"
VAL_CSV = METADATA_DIR / "mixed_val.csv"
TEST_CSV = METADATA_DIR / "mixed_test.csv"

AASIST_PKG_DIR = PROJECT_ROOT / "yubeen" / "aasist"
AASIST_REPO_DIR = PROJECT_ROOT / "aasist"


def _load_module_from_path(module_name, file_path):
    """Load yubeen/aasist/model.py under a private name.

    Both yubeen/ast/ and yubeen/aasist/ have their own model.py, and
    `from train import group_from_mixed_id` above already pulled this
    file's own model.py into sys.modules["model"] as a side effect (train.py
    imports it too) -- a plain `from model import ...` here would silently
    resolve to the wrong one instead of raising. Loading by explicit file
    path sidesteps that name collision.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_aasist_model_mod = _load_module_from_path(
    "_aasist_fake_model", AASIST_PKG_DIR / "model.py",
)
DeepVoiceAASIST = _aasist_model_mod.DeepVoiceAASIST
AASIST_NB_SAMP = _aasist_model_mod.AASIST_MODEL_CONFIG["nb_samp"]

if str(AASIST_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(AASIST_REPO_DIR))
from data_utils import pad as aasist_pad  # noqa: E402 -- original AASIST paper code

DEFAULT_FAKE_CHECKPOINT = AASIST_PKG_DIR / "checkpoints" / "aasist_best.pt"
DEFAULT_PRESENCE_CHECKPOINT = CHECKPOINT_DIR / "presence_best.pt"

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
FAKE_LABELS = ["VF", "MF", "FF"]
PRESENCE_LABELS = ["VP", "MP"]
ALL_LABELS = FAKE_LABELS + PRESENCE_LABELS
SUBMISSION_COLUMNS = [
    "ID",
    "FILE_FAKE_PROB",
    "VOICE_FAKE_PROB",
    "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB",
    "MUSIC_PRESENT_PROB",
]


class PresenceDataset(Dataset):
    """Dataset for voice/music presence labels (VP, MP)."""

    def __init__(self, csv_path, feature_extractor=None):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)

        if feature_extractor is None:
            self.feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_NAME)
        else:
            self.feature_extractor = feature_extractor

        required_columns = ["mixed_id", "split", "VP", "MP"]
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        print(f"Loaded {len(self.df):,} samples from {self.csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # v3 layout: audio/{split}/{mixed_id}.wav
        audio_path = AUDIO_DIR / row["split"] / f"{row['mixed_id']}.wav"
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            raise ValueError(f"Expected 16 kHz but got {sample_rate}: {audio_path}")
        if waveform.shape[0] != 1:
            raise ValueError(f"Expected mono but got {waveform.shape[0]} channels: {audio_path}")

        waveform = waveform.squeeze(0).numpy()
        features = self.feature_extractor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
        )

        input_values = features["input_values"].squeeze(0)
        labels = torch.tensor(
            [float(row["VP"]), float(row["MP"])],
            dtype=torch.float32,
        )

        return {"input_values": input_values, "labels": labels, "audio_path": str(audio_path)}


class PresenceAST(nn.Module):
    """AST backbone + 2-class presence head."""

    def __init__(self, model_name=MODEL_NAME, dropout=0.2, freeze_backbone=True):
        super().__init__()
        self.ast = ASTModel.from_pretrained(model_name)
        hidden_size = self.ast.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for p in self.ast.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.ast.parameters():
            p.requires_grad = True

    def forward(self, input_values):
        outputs = self.ast(input_values=input_values)
        pooled_output = (
            outputs.last_hidden_state[:, 0]
            + outputs.last_hidden_state[:, 1]
        ) / 2
        return self.classifier(pooled_output)


def build_loader(csv_path, batch_size, num_workers, shuffle, limit, seed):
    """Create a loader for presence labels on a split."""
    df = pd.read_csv(csv_path)
    groups = df["mixed_id"].map(group_from_mixed_id)

    if limit is not None:
        df = df.assign(_group=groups)
        per_group = max(1, limit // df["_group"].nunique())
        df = (
            df.groupby("_group", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), per_group), random_state=seed))
            .drop(columns="_group")
            .sort_index()
            .reset_index(drop=True)
        )
        subset_path = Path(csv_path).with_name(f"{Path(csv_path).stem}_presence_{limit}.csv")
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(subset_path, index=False)
        groups = df["mixed_id"].map(group_from_mixed_id)
        csv_path = subset_path

    dataset = PresenceDataset(csv_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader, groups.to_numpy()


def compute_pos_weight(csv_path, limit, seed, device):
    """Compute class weights for VP/MP heads."""
    df = pd.read_csv(csv_path)
    if limit is not None:
        groups = df["mixed_id"].map(group_from_mixed_id)
        df = df.assign(_group=groups)
        per_group = max(1, limit // df["_group"].nunique())
        df = (
            df.groupby("_group", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), per_group), random_state=seed))
            .drop(columns="_group")
            .reset_index(drop=True)
        )

    y = df[PRESENCE_LABELS].to_numpy(dtype=np.float32)
    pos = y.sum(0)
    neg = len(y) - pos
    weight = np.clip(neg / np.maximum(pos, 1.0), 0.1, 10.0)
    return torch.tensor(weight, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate_presence(model, loader, criterion, device, amp_dtype):
    """Presence accuracy plus ROC-AUC per axis -- the AUCs are exactly the
    competition's CPS = 0.5*(Voice Presence ROC-AUC) + 0.5*(Music Presence ROC-AUC)."""
    model.eval()

    all_logits, all_labels = [], []
    total_loss = 0.0
    n = 0

    for batch in loader:
        inputs = batch["input_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(inputs)
            loss = criterion(logits, labels)

        total_loss += loss.item() * inputs.shape[0]
        n += inputs.shape[0]

        all_logits.append(logits.float().cpu())
        all_labels.append(labels.cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = probs >= 0.5

    per_label_acc = {}
    per_label_auc = {}
    for i, name in enumerate(PRESENCE_LABELS):
        per_label_acc[name] = float((preds[:, i] == labels[:, i]).mean())
        per_label_auc[name] = (
            float(roc_auc_score(labels[:, i], probs[:, i]))
            if len(np.unique(labels[:, i])) > 1
            else float("nan")
        )

    cps_like = (
        float(np.mean(list(per_label_auc.values())))
        if not any(np.isnan(list(per_label_auc.values())))
        else float("nan")
    )

    return {
        "loss": total_loss / n,
        "per_label_acc": per_label_acc,
        "per_label_auc": per_label_auc,
        "mean_acc": float(np.mean(list(per_label_acc.values()))),
        "cps_like": cps_like,
    }


def train_presence(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (args.amp and device.type == "cuda") else None
    print(f"device: {device}  amp: {amp_dtype}")

    train_loader, _ = build_loader(
        TRAIN_CSV,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        limit=args.limit,
        seed=args.seed,
    )
    val_loader, _ = build_loader(
        VAL_CSV,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        limit=args.val_limit,
        seed=args.seed,
    )

    model = PresenceAST(dropout=args.dropout, freeze_backbone=True).to(device)
    pos_weight = compute_pos_weight(TRAIN_CSV, args.limit, args.seed, device)
    print("pos_weight:", dict(zip(PRESENCE_LABELS, pos_weight.tolist())))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.classifier.parameters(), "lr": args.head_lr},
            {"params": model.ast.parameters(), "lr": args.backbone_lr},
        ],
        weight_decay=args.weight_decay,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "presence_best.pt"

    best = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        if args.unfreeze_epoch is not None and epoch == args.unfreeze_epoch:
            print(f"--- unfreezing AST backbone at epoch {epoch} ---")
            model.unfreeze_backbone()

        model.train()
        running_loss = 0.0
        seen = 0

        for step, batch in enumerate(train_loader):
            inputs = batch["input_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(inputs)
                loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * inputs.shape[0]
            seen += inputs.shape[0]

            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)}  loss {running_loss / seen:.4f}", flush=True)

        metrics = evaluate_presence(model, val_loader, criterion, device, amp_dtype)
        cps_like = metrics["cps_like"]
        print(
            f"[epoch {epoch}] train_loss {running_loss / seen:.4f}  "
            f"val_loss {metrics['loss']:.4f}  "
            f"VP_auc {metrics['per_label_auc']['VP']:.3f}  "
            f"MP_auc {metrics['per_label_auc']['MP']:.3f}  "
            f"cps_like {cps_like:.3f}",
            flush=True,
        )

        history.append({"epoch": epoch, "train_loss": running_loss / seen, **metrics})

        if cps_like > best:
            best = cps_like
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

    print(f"\nbest val cps_like: {best:.4f}")
    result = {"best_val_cps_like": best, "history": history}

    if getattr(args, "eval_test", False):
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        test_loader, _ = build_loader(
            TEST_CSV,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            limit=None,
            seed=args.seed,
        )
        test_metrics = evaluate_presence(model, test_loader, criterion, device, amp_dtype)
        result["test"] = test_metrics
        print("\n=== test ===")
        print(json.dumps(test_metrics, indent=2))

    (out_dir / "presence_training_report.json").write_text(json.dumps(result, indent=2))


@torch.no_grad()
def evaluate_saved_checkpoint(checkpoint_path, split, batch_size, num_workers, device, amp_enabled):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    dropout = saved_args.get("dropout", 0.2)
    seed = saved_args.get("seed", 42)

    model = PresenceAST(dropout=dropout, freeze_backbone=False).to(device)
    model.load_state_dict(checkpoint["model"])

    pos_weight = compute_pos_weight(TRAIN_CSV, None, seed, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    amp_dtype = torch.bfloat16 if amp_enabled and device.type == "cuda" else None

    split_map = {"val": VAL_CSV, "test": TEST_CSV}
    loader, _ = build_loader(
        split_map[split],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        limit=None,
        seed=seed,
    )

    metrics = evaluate_presence(model, loader, criterion, device, amp_dtype)
    return {"checkpoint": str(checkpoint_path), "split": split, "metrics": metrics}


def extract_audio_features(audio_path, feature_extractor, device):
    """Convert a single audio file to an AST input tensor (for presence)."""
    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)

    features = feature_extractor(
        waveform.squeeze(0).numpy(),
        sampling_rate=16000,
        return_tensors="pt",
    )
    return features["input_values"].to(device)


def extract_aasist_features(audio_path, device):
    """Convert a single audio file to an AASIST input tensor (for VF/MF/FF).

    Uses the original AASIST paper's deterministic `pad` (not the random
    training-time crop) so a given file always scores the same way here.
    """
    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)

    x = aasist_pad(waveform.squeeze(0).numpy(), AASIST_NB_SAMP)
    return torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)


def predict_5_heads(audio_path, fake_model=None, presence_model=None, threshold=0.5, device=None):
    """Predict all 5 heads on a single real audio file.

    VF/MF/FF come from an AASIST fake-detector, VP/MP from an AST presence
    head -- two different models, two different audio representations, one
    combined result.

    Output format:
      {
        "VF": 0.12,
        "MF": 0.88,
        "FF": 0.91,
        "VP": 0.97,
        "MP": 0.14,
        "decision": {"VF": False, ...}
      }
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if fake_model is None:
        fake_model = DeepVoiceAASIST(dropout=0.2, freeze_backbone=False).to(device)
    if presence_model is None:
        presence_model = PresenceAST(dropout=0.2, freeze_backbone=False).to(device)

    fake_model.eval()
    presence_model.eval()

    feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_NAME)
    ast_inputs = extract_audio_features(audio_path, feature_extractor, device)
    aasist_inputs = extract_aasist_features(audio_path, device)

    with torch.no_grad():
        fake_logits = fake_model(aasist_inputs)
        presence_logits = presence_model(ast_inputs)

    fake_probs = torch.sigmoid(fake_logits).squeeze(0).cpu().numpy()
    presence_probs = torch.sigmoid(presence_logits).squeeze(0).cpu().numpy()

    results = {}
    for label, prob in zip(FAKE_LABELS, fake_probs):
        results[label] = float(prob)
    for label, prob in zip(PRESENCE_LABELS, presence_probs):
        results[label] = float(prob)

    results["decision"] = {
        label: bool(prob >= threshold)
        for label, prob in results.items()
        if label in ALL_LABELS
    }
    return results


def predict_audio_dir(audio_dir, fake_checkpoint, presence_checkpoint, threshold=0.5, device=None):
    """Evaluate every audio file in a directory and print combined 5-head results."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fake_model = DeepVoiceAASIST(dropout=0.2, freeze_backbone=False, pretrained=False).to(device)
    if fake_checkpoint.exists():
        state = torch.load(fake_checkpoint, map_location=device, weights_only=False)
        fake_model.load_state_dict(state["model"])
    else:
        raise FileNotFoundError(f"Fake checkpoint not found: {fake_checkpoint}")

    presence_model = PresenceAST(dropout=0.2, freeze_backbone=False).to(device)
    if presence_checkpoint.exists():
        state = torch.load(presence_checkpoint, map_location=device, weights_only=False)
        presence_model.load_state_dict(state["model"])
    else:
        raise FileNotFoundError(f"Presence checkpoint not found: {presence_checkpoint}")

    output = []
    for path in sorted(Path(audio_dir).glob("**/*")):
        if path.suffix.lower() not in {".wav", ".flac", ".mp3"}:
            continue
        result = predict_5_heads(str(path), fake_model=fake_model, presence_model=presence_model, threshold=threshold, device=device)
        result["audio_path"] = str(path)
        result["ID"] = path.stem
        output.append(result)

    return output


def build_submission_dataframe(results):
    """Convert 5-head result list into the challenge submission CSV format."""
    rows = []
    for item in results:
        row = {
            "ID": item.get("ID") or Path(item["audio_path"]).stem,
            "FILE_FAKE_PROB": float(item["FF"]),
            "VOICE_FAKE_PROB": float(item["VF"]),
            "MUSIC_FAKE_PROB": float(item["MF"]),
            "VOICE_PRESENT_PROB": float(item["VP"]),
            "MUSIC_PRESENT_PROB": float(item["MP"]),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=SUBMISSION_COLUMNS)


def write_submission_csv(audio_dir, output_path, fake_checkpoint, presence_checkpoint, threshold=0.5, device=None):
    """Create a Dacon-style submission CSV from a directory of audio files."""
    results = predict_audio_dir(
        audio_dir,
        fake_checkpoint=Path(fake_checkpoint),
        presence_checkpoint=Path(presence_checkpoint),
        threshold=threshold,
        device=device,
    )
    df = build_submission_dataframe(results)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"saved submission csv -> {out_path}")
    return df


def resolve_checkpoint(explicit_path, filename, project_root=PROJECT_ROOT):
    """Resolve checkpoint path automatically without manual path edits."""
    if explicit_path is not None:
        p = Path(explicit_path)
        if p.exists():
            return p
        if not p.is_absolute():
            candidates = [
                project_root / p,
                project_root / "checkpoints" / p,
                project_root / "yubeen" / "ast" / p,
                project_root / "yubeen" / "aasist" / p,
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
        return p

    candidates = [
        project_root / "checkpoints" / filename,
        project_root / "yubeen" / "ast" / "checkpoints" / filename,
        project_root / "yubeen" / "aasist" / "checkpoints" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return project_root / "checkpoints" / filename


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--unfreeze-epoch", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="checkpoints")
    parser.add_argument("--eval-test", action="store_true",
                        help="After training, evaluate the best checkpoint on "
                             "the test split too (train_presence() checks this "
                             "but nothing used to set it -- test metrics never "
                             "made it into presence_training_report.json).")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default="checkpoints/presence_best.pt")
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")

    parser.add_argument("--audio", type=str, default=None, help="Single audio file to test with the 5-head model")
    parser.add_argument("--audio-dir", type=str, default=None, help="Directory containing audio files to test")
    parser.add_argument("--fake-checkpoint", type=str, default=str(DEFAULT_FAKE_CHECKPOINT))
    parser.add_argument("--presence-checkpoint", type=str, default=str(DEFAULT_PRESENCE_CHECKPOINT))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--submission-out", type=str, default=None, help="Path for Dacon-style submission CSV")

    args = parser.parse_args()
    args.fake_checkpoint = str(resolve_checkpoint(args.fake_checkpoint, "aasist_best.pt"))
    args.presence_checkpoint = str(resolve_checkpoint(args.presence_checkpoint, "presence_best.pt"))

    if args.audio is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fake_model = DeepVoiceAASIST(dropout=0.2, freeze_backbone=False, pretrained=False).to(device)
        fake_state = torch.load(args.fake_checkpoint, map_location=device, weights_only=False)
        fake_model.load_state_dict(fake_state["model"])

        presence_model = PresenceAST(dropout=0.2, freeze_backbone=False).to(device)
        presence_state = torch.load(args.presence_checkpoint, map_location=device, weights_only=False)
        presence_model.load_state_dict(presence_state["model"])

        result = predict_5_heads(
            args.audio,
            fake_model=fake_model,
            presence_model=presence_model,
            threshold=args.threshold,
            device=device,
        )
        result["audio_path"] = args.audio
        print(json.dumps(result, indent=2))

        if args.json_out is not None:
            Path(args.json_out).write_text(json.dumps(result, indent=2))
        return

    if args.audio_dir is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = predict_audio_dir(
            args.audio_dir,
            fake_checkpoint=Path(args.fake_checkpoint),
            presence_checkpoint=Path(args.presence_checkpoint),
            threshold=args.threshold,
            device=device,
        )
        print(json.dumps(results, indent=2))

        if args.json_out is not None:
            Path(args.json_out).write_text(json.dumps(results, indent=2))

        if args.submission_out is not None:
            df = build_submission_dataframe(results)
            out_path = Path(args.submission_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            print(f"saved submission csv -> {out_path}")
        return

    if args.eval_only:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        saved_amp = torch.load(checkpoint_path, map_location="cpu", weights_only=False).get("args", {}).get("amp", False)
        amp_enabled = saved_amp if args.amp is None else args.amp

        splits = ["val", "test"] if args.split == "both" else [args.split]
        results = {}
        for split in splits:
            result = evaluate_saved_checkpoint(
                checkpoint_path,
                split,
                args.batch_size,
                args.num_workers,
                device,
                amp_enabled,
            )
            results[split] = result
            print(json.dumps(result, indent=2))

        report_path = Path(args.out) / "presence_evaluation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(results, indent=2))
        print(f"saved evaluation report -> {report_path}")
        return

    train_presence(args)


if __name__ == "__main__":
    main()
