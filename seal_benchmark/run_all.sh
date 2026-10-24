#!/usr/bin/env bash
# Full pipeline: manifest -> train both models x 3 seeds -> evaluate -> aggregate.
set -euo pipefail

IMAGES=dataset/images
MODELS="mobilenetv3_small_100 efficientnet_b0"
SEEDS="0 1 2"
HOLDOUT_PHONES="D"             # phone kept unseen in training; "" to disable

python make_manifest.py --images "$IMAGES" --splits splits.json --holdout-phones "$HOLDOUT_PHONES" --out manifest.csv
for m in $MODELS; do
  for s in $SEEDS; do
    python train.py    --model "$m" --seed "$s" --images "$IMAGES" --manifest manifest.csv
    python evaluate.py --run "runs/${m}_seed${s}"
  done
done
python aggregate.py
