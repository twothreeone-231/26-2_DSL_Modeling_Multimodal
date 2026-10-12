from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import ASTFeatureExtractor


# Dynamically derived (matches train.py/evaluation.py) instead of a hardcoded
# /data1/... string: the login node sees this shared storage at /data1/...,
# but compute nodes only see it at /mnt/data1/... (a hardcoded /data1/ path
# here silently pointed at a different, unrelated local /data1 on the compute
# node and caused FileNotFoundError on every audio file during training).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "yusung/data_mixed_v5/audio"

FAKE_LABELS = ["VF", "MF", "FF"]


class MixedASTDataset(Dataset):
    """
    Dataset for the v3 mixed voice + music samples.

    Targets
    -------
    VF : Voice Fake
    MF : Music Fake
    FF : File Fake (== VF or MF, restricted to components that are present)

    Unlike v2, a v3 clip does not always contain both a voice and a music
    component (there are voice_only / music_only / noise_only groups too).
    When a component is absent, its label is -1 in the CSV -- there is no
    ground truth for it, matching the competition's own rule that Voice EER
    / Music EER are only computed on samples where that component exists.
    Those rows get mask=0 for that axis so BCE loss and metrics skip them
    instead of training against a fabricated 0/1 target.
    """

    def __init__(
        self,
        csv_path,
        feature_extractor=None,
    ):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)

        if feature_extractor is None:
            self.feature_extractor = ASTFeatureExtractor.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593"
            )
        else:
            self.feature_extractor = feature_extractor

        required_columns = ["mixed_id", "split", *FAKE_LABELS]

        missing = [
            col for col in required_columns
            if col not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        print(f"Loaded {len(self.df):,} samples from {self.csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # v3 layout: audio/{split}/{mixed_id}.wav
        audio_path = DATA_ROOT / row["split"] / f"{row['mixed_id']}.wav"

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        waveform, sample_rate = torchaudio.load(audio_path)

        # v3 preprocessing should already be:
        # mono / 16 kHz / 10 sec
        # These checks detect preprocessing problems instead of
        # silently preprocessing the file again.
        if sample_rate != 16000:
            raise ValueError(
                f"Expected 16 kHz but got {sample_rate}: {audio_path}"
            )

        if waveform.shape[0] != 1:
            raise ValueError(
                f"Expected mono but got {waveform.shape[0]} channels: "
                f"{audio_path}"
            )

        waveform = waveform.squeeze(0).numpy()

        features = self.feature_extractor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
        )

        # (1, time, mel) -> (time, mel)
        input_values = features["input_values"].squeeze(0)

        raw = row[FAKE_LABELS].to_numpy(dtype="float32")
        mask = torch.tensor((raw != -1).astype("float32"))
        labels = torch.tensor(raw, dtype=torch.float32)
        labels[mask == 0] = 0.0  # placeholder value; loss/metrics ignore it via mask

        return {
            "input_values": input_values,
            "labels": labels,
            "mask": mask,
            "audio_path": str(audio_path),
        }
