#!/usr/bin/env bash
set -euo pipefail
COCO=${1:-$(pwd)/data/coco}
VQA=${2:-$(pwd)/data/vqav2}
mkdir -p "$VQA/images"
ln -snf "$COCO/val2014" "$VQA/images/val2014"
ln -snf "$COCO/train2014" "$VQA/images/train2014"