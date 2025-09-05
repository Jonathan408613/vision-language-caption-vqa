#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:-$(pwd)/data/coco}
mkdir -p "$ROOT" && cd "$ROOT"


# Images
wget -c http://images.cocodataset.org/zips/train2014.zip
wget -c http://images.cocodataset.org/zips/val2014.zip
unzip -n train2014.zip
unzip -n val2014.zip


# Captions annotations
wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip -n annotations_trainval2014.zip