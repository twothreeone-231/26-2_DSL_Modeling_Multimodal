"""Raw-waveform dataset for AASIST on the v3 mixed voice + music samples.

Mirrors yubeen/ast/dataset.py's VF/MF/FF masking (voice_only / music_only /
noise_only groups leave the missing axis's label -1 in the CSV, with
mask=0 so loss/metrics skip it) but returns a fixed-length raw waveform
instead of an AST log-mel feature, since AASIST's SincConv front-end reads
the signal directly.

`pad`/`pad_random` are imported straight from the original AASIST paper's
repo (deepvoice/aasist/data_utils.py) rather than reimplemented, so a train
clip gets the same random 4.04s (64600-sample) crop the paper's own
training loop used as augmentation, and a val/test clip gets the same
deterministic crop (from the start, tiled if the clip were shorter) so
evaluation is reproducible.
"""

import sys
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Same /data1 vs /mnt/data1 note as yubeen/ast/dataset.py: derived from
# __file__ so this resolves correctly on both the login node and compute
# nodes instead of hardcoding one or the other.
DATA_ROOT = PROJECT_ROOT / "yusung/data_mixed_v5/audio"

AASIST_REPO_ROOT = PROJECT_ROOT / "aasist"
if str(AASIST_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(AASIST_REPO_ROOT))

from data_utils import pad, pad_random  # noqa: E402

FAKE_LABELS = ["VF", "MF", "FF"]
NB_SAMP = 64600  # ~4.04s at 16kHz -- matches the pretrained AASIST.pth


class MixedAASISTDataset(Dataset):
    """
    Dataset for the v3 mixed voice + music samples, AASIST side.

    Targets are identical to yubeen/ast/dataset.py's MixedASTDataset (VF,
    MF, FF with -1/mask for an absent component) -- only the audio
    representation differs.
    """

    def __init__(self, csv_path, random_crop=False):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.random_crop = random_crop

        required_columns = ["mixed_id", "split", *FAKE_LABELS]
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        print(f"Loaded {len(self.df):,} samples from {self.csv_path} "
              f"(random_crop={self.random_crop})")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # v3 layout: audio/{split}/{mixed_id}.wav
        audio_path = DATA_ROOT / row["split"] / f"{row['mixed_id']}.wav"

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform, sample_rate = torchaudio.load(audio_path)

        if sample_rate != 16000:
            raise ValueError(
                f"Expected 16 kHz but got {sample_rate}: {audio_path}"
            )
        if waveform.shape[0] != 1:
            raise ValueError(
                f"Expected mono but got {waveform.shape[0]} channels: "
                f"{audio_path}"
            )

        x = waveform.squeeze(0).numpy()
        x = pad_random(x, NB_SAMP) if self.random_crop else pad(x, NB_SAMP)
        input_values = torch.tensor(x, dtype=torch.float32)

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
