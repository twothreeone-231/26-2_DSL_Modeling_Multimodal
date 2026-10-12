"""Dacon evaluation server entrypoint.

Self-contained on purpose: the submit.zip layout is fixed to
model/ + script.py + requirements.txt, so no sibling .py modules are
imported here. Everything (model definitions, audio loading, inference,
CSV writing) lives in this single file.

Expected layout when this runs on the eval server:
  ./model/ast_backbone/{config.json,model.safetensors,preprocessor_config.json}
  ./model/ast_best.pt
  ./model/presence_best.pt
  ./data/test/*.{wav,mp3,flac}      (read-only, injected by the server)
  ./data/sample_submission.csv      (read-only, injected by the server)
  ./output/submission.csv           (must be created by this script)
"""

import os
import sys
import warnings
from pathlib import Path

# Inference must only use local files under model/ - never hit the network.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR / "model"
AST_BACKBONE_DIR = MODEL_DIR / "ast_backbone"
FAKE_CHECKPOINT = MODEL_DIR / "ast_best.pt"
PRESENCE_CHECKPOINT = MODEL_DIR / "presence_best.pt"

TEST_DIR = SCRIPT_DIR / "data" / "test"
SAMPLE_SUBMISSION = SCRIPT_DIR / "data" / "sample_submission.csv"
OUTPUT_PATH = SCRIPT_DIR / "output" / "submission.csv"

AUDIO_EXTS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"}
TARGET_SR = 16000

FAKE_LABELS = ["VF", "MF", "FF"]
PRESENCE_LABELS = ["VP", "MP"]

SUBMISSION_COLUMN_MAP = {
    "FILE_FAKE_PROB": "FF",
    "VOICE_FAKE_PROB": "VF",
    "MUSIC_FAKE_PROB": "MF",
    "VOICE_PRESENT_PROB": "VP",
    "MUSIC_PRESENT_PROB": "MP",
}


# --------------------------------------------------------------------------
# Model definitions (mirrors model.py / presence.py, kept inline for a
# single-file, offline-loadable submission)
# --------------------------------------------------------------------------

class DeepVoiceAST(nn.Module):
    """AST backbone + 3-class fake-detection head (VF / MF / FF)."""

    def __init__(self, backbone_path, dropout=0.2):
        super().__init__()
        from transformers import ASTModel

        self.ast = ASTModel.from_pretrained(str(backbone_path))
        hidden_size = self.ast.config.hidden_size
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 3))

    def forward(self, input_values):
        outputs = self.ast(input_values=input_values)
        pooled = (outputs.last_hidden_state[:, 0] + outputs.last_hidden_state[:, 1]) / 2
        return self.classifier(pooled)


class PresenceAST(nn.Module):
    """AST backbone + 2-class presence head (VP / MP)."""

    def __init__(self, backbone_path, dropout=0.2):
        super().__init__()
        from transformers import ASTModel

        self.ast = ASTModel.from_pretrained(str(backbone_path))
        hidden_size = self.ast.config.hidden_size
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 2))

    def forward(self, input_values):
        outputs = self.ast(input_values=input_values)
        pooled = (outputs.last_hidden_state[:, 0] + outputs.last_hidden_state[:, 1]) / 2
        return self.classifier(pooled)


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------

def select_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_waveform(path):
    """Load an audio file as mono float32 @ 16kHz, robust to wav/mp3/flac."""
    import librosa

    waveform, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
    return waveform.astype(np.float32)


def find_audio_files(test_dir):
    files = [p for p in Path(test_dir).rglob("*") if p.suffix.lower() in AUDIO_EXTS]
    if not files:
        raise FileNotFoundError(f"No audio files found under {test_dir}")
    return files


def read_sample_submission(sample_submission_path):
    df = pd.read_csv(sample_submission_path)
    id_col = df.columns[0]
    return list(df.columns), df[id_col].astype(str).tolist()


def order_audio_files(audio_files, ordered_ids):
    """Reorder audio_files to match the ID order in sample_submission.csv."""
    by_stem = {p.stem: p for p in audio_files}
    ordered = []
    missing = []
    for file_id in ordered_ids:
        path = by_stem.get(file_id)
        if path is None:
            missing.append(file_id)
            ordered.append(None)
        else:
            ordered.append(path)
    if missing:
        raise FileNotFoundError(f"{len(missing)} IDs from sample_submission.csv have no matching audio file, e.g. {missing[:5]}")
    return ordered


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------

def load_models(device):
    from transformers import ASTFeatureExtractor

    feature_extractor = ASTFeatureExtractor.from_pretrained(str(AST_BACKBONE_DIR))

    fake_model = DeepVoiceAST(AST_BACKBONE_DIR).to(device)
    fake_state = torch.load(FAKE_CHECKPOINT, map_location=device, weights_only=False)
    fake_model.load_state_dict(fake_state["model"])
    fake_model.eval()

    presence_model = PresenceAST(AST_BACKBONE_DIR).to(device)
    presence_state = torch.load(PRESENCE_CHECKPOINT, map_location=device, weights_only=False)
    presence_model.load_state_dict(presence_state["model"])
    presence_model.eval()

    return feature_extractor, fake_model, presence_model


@torch.no_grad()
def predict_one(path, feature_extractor, fake_model, presence_model, device):
    waveform = load_waveform(path)
    features = feature_extractor(waveform, sampling_rate=TARGET_SR, return_tensors="pt")
    input_values = features["input_values"].to(device)

    fake_probs = torch.sigmoid(fake_model(input_values)).squeeze(0).cpu().numpy()
    presence_probs = torch.sigmoid(presence_model(input_values)).squeeze(0).cpu().numpy()

    result = dict(zip(FAKE_LABELS, fake_probs.tolist()))
    result.update(dict(zip(PRESENCE_LABELS, presence_probs.tolist())))
    return result


def run_inference(audio_files, ids, feature_extractor, fake_model, presence_model, device):
    rows = []
    for i, (file_id, path) in enumerate(zip(ids, audio_files)):
        try:
            scores = predict_one(path, feature_extractor, fake_model, presence_model, device)
        except Exception as exc:  # noqa: BLE001 - one bad file must not sink the whole run
            print(f"[warn] inference failed for {file_id} ({path}): {exc}", file=sys.stderr)
            scores = {label: 0.5 for label in FAKE_LABELS + PRESENCE_LABELS}
        rows.append({"ID": file_id, **scores})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(ids)} done", flush=True)
    return rows


def build_submission(column_names, rows):
    df = pd.DataFrame(rows).set_index("ID")
    out = pd.DataFrame(index=df.index)
    for col in column_names:
        if col in SUBMISSION_COLUMN_MAP:
            out[col] = df[SUBMISSION_COLUMN_MAP[col]].astype(float)
        else:
            out[col] = df.index  # the ID column itself
    out = out[column_names].reset_index(drop=True)
    return out


def main():
    device = select_device()
    print(f"device: {device}")

    audio_files = find_audio_files(TEST_DIR)
    column_names, ids = read_sample_submission(SAMPLE_SUBMISSION)
    ordered_files = order_audio_files(audio_files, ids)

    feature_extractor, fake_model, presence_model = load_models(device)

    rows = run_inference(ordered_files, ids, feature_extractor, fake_model, presence_model, device)

    submission = build_submission(column_names, rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(submission)} predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
