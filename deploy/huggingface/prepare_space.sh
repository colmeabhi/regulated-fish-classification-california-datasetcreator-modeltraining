#!/bin/bash
# Assemble a Hugging Face Space folder from this repo + your trained models.
# Usage: ./deploy/huggingface/prepare_space.sh [output_dir]   (default: hf_space/)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:-$ROOT/hf_space}"

for f in hierarchy_config.json regulatory_classifier.pth \
         species_classifier_prohibited.pth species_classifier_restricted.pth species_classifier_legal.pth; do
    if [ ! -f "$ROOT/models/$f" ]; then
        echo "Missing models/$f - run scripts/10_train_hierarchical_classification.py first" >&2
        exit 1
    fi
done

mkdir -p "$OUT/models"
cp "$ROOT/deploy/huggingface/app.py" "$ROOT/deploy/huggingface/requirements.txt" "$OUT/"
cp "$ROOT/deploy/huggingface/SPACE_README.md" "$OUT/README.md"
cp "$ROOT"/models/hierarchy_config.json "$ROOT"/models/*.pth "$OUT/models/"

echo "Space ready in $OUT"
echo "Test locally:   cd $OUT && python app.py"
