"""Shared training/eval utilities for the v3 mixed dataset.

Used by both yubeen/ast (AST backbone: VP/MP presence, plus the legacy
VF/MF/FF classifier in ast/train.py) and yubeen/aasist (AASIST backbone:
VF/MF/FF fake detection, replacing AST for that half of the pipeline). The
two model-specific train.py scripts only differ in the model itself and how
audio is turned into `input_values` -- the masked-BCE loss, EER, group-
balanced subset sampling, and the VF/MF/FF eval metrics are identical
either way, so they live here once.
"""

import re

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve


GROUP_ORDER = [
    "voice_only",
    "voice_fake_only",
    "music_only",
    "music_fake_only",
    "voice_music",
    "voice_music_fake",
    "voice_fake_music",
    "voice_fake_music_fake",
    "noise_only",
]

# mixed_id looks like "<group>_<split>_<index>", e.g.
# "voice_fake_music_fake_train_003995" -> group "voice_fake_music_fake".
# v4/v5 added SONICS-sourced rows in voice_fake_music_fake, whose mixed_id
# has an extra "_sonics" before the index (e.g.
# "voice_fake_music_fake_train_sonics_000430") -- the optional group below
# absorbs that so it doesn't get swallowed into the captured group name.
_GROUP_RE = re.compile(r"^(.*)_(train|val|test)(?:_sonics)?_\d+$")


def group_from_mixed_id(mixed_id):
    match = _GROUP_RE.match(mixed_id)
    if not match:
        raise ValueError(f"Unrecognized mixed_id format: {mixed_id}")
    return match.group(1)


def balanced_subset(df, limit, seed):
    """Sample up to `limit` rows, balanced across the 9 label groups."""
    groups = df["mixed_id"].map(group_from_mixed_id)
    group_names = groups.unique()
    per_group = max(1, limit // len(group_names))
    rng = np.random.default_rng(seed)

    parts = []
    for name in group_names:
        idx = df.index[groups == name].to_numpy()
        n = min(len(idx), per_group)
        parts.append(rng.choice(idx, size=n, replace=False))

    keep = np.sort(np.concatenate(parts))
    return df.loc[keep].reset_index(drop=True)


def compute_masked_pos_weight(df, label_names, device):
    """Per-axis class weight, counting only rows where that axis has a
    label -- others are the -1 sentinel used when voice/music is absent
    from the clip, and should not skew the weight."""
    y = df[label_names].to_numpy(dtype=np.float32)
    valid = y != -1
    pos = np.where(valid, y == 1, False).sum(0).astype(np.float32)
    neg = np.where(valid, y == 0, False).sum(0).astype(np.float32)
    weight = np.clip(neg / np.maximum(pos, 1.0), 0.1, 10.0)
    return torch.tensor(weight, dtype=torch.float32, device=device)


def masked_bce_loss(logits, labels, mask, pos_weight):
    """BCE-with-logits averaged only over (label, mask) pairs where mask == 1."""
    per_element = F.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=pos_weight, reduction="none",
    )
    per_element = per_element * mask
    denom = mask.sum().clamp(min=1.0)
    return per_element.sum() / denom


def compute_eer(y_true, y_score):
    """Equal Error Rate, mirroring the competition's own definition:
    FAKE is the positive class (1); EER is the FPR/FNR crossing point."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


@torch.no_grad()
def evaluate_masked_multilabel(model, loader, groups, criterion, device, amp_dtype,
                               label_names=("VF", "MF", "FF")):
    """VF/MF/FF eval shared by the AST and AASIST fake-detectors: per-label
    acc/EER, the ads_like aggregate (0.5*(1-FileEER) + 0.2*(1-VoiceEER) +
    0.3*(1-MusicEER)) that mirrors the competition score, and per-group FF
    accuracy across the 9 label groups."""
    model.eval()

    all_logits, all_labels, all_masks = [], [], []
    total_loss = 0.0
    n = 0

    for batch in loader:
        inputs = batch["input_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            logits = model(inputs)
            loss = criterion(logits, labels, mask)

        total_loss += loss.item() * inputs.shape[0]
        n += inputs.shape[0]

        all_logits.append(logits.float().cpu())
        all_labels.append(labels.cpu())
        all_masks.append(mask.cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    masks = torch.cat(all_masks).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = probs >= 0.5

    per_label_acc = {}
    per_label_eer = {}
    for i, name in enumerate(label_names):
        valid = masks[:, i] == 1
        if valid.sum() == 0:
            per_label_acc[name] = float("nan")
            per_label_eer[name] = float("nan")
            continue
        per_label_acc[name] = float((preds[valid, i] == labels[valid, i]).mean())
        per_label_eer[name] = compute_eer(labels[valid, i], probs[valid, i])

    file_eer = per_label_eer["FF"]
    voice_eer = per_label_eer["VF"]
    music_eer = per_label_eer["MF"]
    ads_like = (
        0.5 * (1 - file_eer) + 0.2 * (1 - voice_eer) + 0.3 * (1 - music_eer)
        if not any(np.isnan([file_eer, voice_eer, music_eer]))
        else float("nan")
    )

    per_group_ff_acc = {}
    ff_idx = label_names.index("FF")
    for name in GROUP_ORDER:
        sel = groups == name
        if sel.any():
            per_group_ff_acc[name] = float(
                (preds[sel, ff_idx] == labels[sel, ff_idx]).mean()
            )

    return {
        "loss": total_loss / n,
        "per_label_acc": per_label_acc,
        "per_label_eer": per_label_eer,
        "ads_like": ads_like,
        "per_group_ff_acc": per_group_ff_acc,
    }
