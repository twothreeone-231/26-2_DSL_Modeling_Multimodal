"""DeepVoiceAASIST: the AASIST encoder + a 3-head VF/MF/FF classifier.

AASIST (Jung et al., 2022, clovaai/aasist -- cloned as-is at
deepvoice/aasist so this wraps the original paper code rather than
reimplementing it) was built for binary bonafide/spoof detection on raw
16 kHz waveforms: a SincConv front-end, a residual encoder, and two graph
attention branches (spectral + temporal) pooled into a single 160-dim
embedding, finished off with `nn.Linear(160, 2)`.

Here we keep everything up to that embedding (`last_hidden`) and swap the
2-class head for the same kind of multi-label VF/MF/FF classifier
yubeen/ast/model.py uses for AST, so training (yubeen/common.py's masked
BCE / EER code) is identical between the two backbones. The pretrained
encoder weights (models/weights/AASIST.pth in the cloned repo) are loaded
by default -- ASVspoof-style bonafide/spoof discrimination is a reasonable
starting point for our VF/MF/FF task even though the output head differs.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AASIST_REPO_ROOT = PROJECT_ROOT / "aasist"

if str(AASIST_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(AASIST_REPO_ROOT))

from models.AASIST import Model as AASISTBackbone  # noqa: E402

# Matches aasist/config/AASIST.conf's "model_config" (the full AASIST, not
# the lighter AASIST-L) so the shapes match the pretrained AASIST.pth below.
AASIST_MODEL_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}
PRETRAINED_WEIGHTS = AASIST_REPO_ROOT / "models" / "weights" / "AASIST.pth"

EMBEDDING_SIZE = 5 * AASIST_MODEL_CONFIG["gat_dims"][1]  # 160, see Model.out_layer


class DeepVoiceAASIST(nn.Module):

    def __init__(
        self,
        dropout=0.2,
        freeze_backbone=True,
        pretrained=True,
    ):
        super().__init__()

        self.backbone = AASISTBackbone(AASIST_MODEL_CONFIG)
        if pretrained and PRETRAINED_WEIGHTS.exists():
            state = torch.load(PRETRAINED_WEIGHTS, map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(state)

        # VF / MF / FF classifier, on top of the backbone's pooled
        # embedding. The backbone's own bonafide/spoof out_layer stays
        # (harmless, unused parameters) so the pretrained state_dict above
        # loads without key surgery.
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(EMBEDDING_SIZE, 3),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, input_values):
        last_hidden, _unused_bonafide_spoof_logits = self.backbone(input_values)
        return self.classifier(last_hidden)
