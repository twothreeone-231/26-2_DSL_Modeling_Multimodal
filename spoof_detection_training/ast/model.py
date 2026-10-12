import torch
import torch.nn as nn
from transformers import ASTModel


MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"


class DeepVoiceAST(nn.Module):

    def __init__(
        self,
        model_name=MODEL_NAME,
        dropout=0.2,
        freeze_backbone=True,
    ):
        super().__init__()

        # Pretrained AST backbone
        self.ast = ASTModel.from_pretrained(model_name)

        hidden_size = self.ast.config.hidden_size

        # VF / MF / FF classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.ast.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.ast.parameters():
            param.requires_grad = True

    def forward(self, input_values):
        outputs = self.ast(input_values=input_values)

        pooled_output = (
            outputs.last_hidden_state[:, 0]
            + outputs.last_hidden_state[:, 1]
        ) / 2

        logits = self.classifier(pooled_output)

        return logits