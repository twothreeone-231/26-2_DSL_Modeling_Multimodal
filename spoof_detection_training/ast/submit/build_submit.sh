#!/usr/bin/env bash
# Assembles submit/model/ and zips submit.zip in the layout Dacon requires:
#   submit.zip
#   ├── model/
#   │     ├── ast_backbone/{config.json,model.safetensors,preprocessor_config.json}
#   │     ├── ast_best.pt
#   │     └── presence_best.pt
#   ├── script.py
#   └── requirements.txt
#
# Usage:
#   ./build_submit.sh <ast_best.pt path> <presence_best.pt path>
set -euo pipefail

FAKE_CKPT="${1:?usage: build_submit.sh <ast_best.pt> <presence_best.pt>}"
PRESENCE_CKPT="${2:?usage: build_submit.sh <ast_best.pt> <presence_best.pt>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/model"
BACKBONE_DIR="$MODEL_DIR/ast_backbone"

HF_SNAPSHOT=$(find ~/.cache/huggingface/hub/models--MIT--ast-finetuned-audioset-10-10-0.4593/snapshots -mindepth 1 -maxdepth 1 -type d | head -1)
if [[ -z "$HF_SNAPSHOT" ]]; then
    echo "AST backbone not found in the HF cache. Run once online first to populate it:" >&2
    echo '  python -c "from transformers import ASTModel, ASTFeatureExtractor; m=\"MIT/ast-finetuned-audioset-10-10-0.4593\"; ASTModel.from_pretrained(m); ASTFeatureExtractor.from_pretrained(m)"' >&2
    exit 1
fi

rm -rf "$MODEL_DIR"
mkdir -p "$BACKBONE_DIR"
for f in config.json model.safetensors preprocessor_config.json; do
    cp -L "$HF_SNAPSHOT/$f" "$BACKBONE_DIR/$f"
done

cp "$FAKE_CKPT" "$MODEL_DIR/ast_best.pt"
cp "$PRESENCE_CKPT" "$MODEL_DIR/presence_best.pt"

cd "$SCRIPT_DIR"
rm -f submit.zip
zip -r submit.zip model script.py requirements.txt -x '*.DS_Store'

echo "Built $SCRIPT_DIR/submit.zip"
du -sh "$SCRIPT_DIR/submit.zip"
