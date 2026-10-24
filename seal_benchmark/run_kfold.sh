#!/usr/bin/env bash
# Grouped k-fold: manifest -> folds -> train + evaluate every (fold, model, seed) -> aggregate.
# Run the single-split pilot first (run_all.sh or the commands in the README).
set -euo pipefail

IMAGES=dataset/images
K=5
MODELS="mobilenetv3_small_100 efficientnet_b0"
SEEDS="0"                       # one seed per fold is enough; folds already vary the box choice
HOLDOUT_PHONES="D"              # phone(s) kept unseen in training; "" to disable
TRAIN_ARGS=""                   # e.g. "--stage2-epochs 30 --patience 5" (decide after checking the pilot's history.csv)

python make_manifest.py --images "$IMAGES" --splits splits.json --out manifest.csv
python make_folds.py --manifest manifest.csv --folds "$K" --holdout-phones "$HOLDOUT_PHONES" --out folds
for f in $(seq 0 $((K-1))); do
  for m in $MODELS; do
    for s in $SEEDS; do
      python train.py --model "$m" --seed "$s" --images "$IMAGES" \
             --manifest "folds/fold${f}/manifest.csv" --out "runs_kfold/fold${f}" $TRAIN_ARGS
      python evaluate.py --run "runs_kfold/fold${f}/${m}_seed${s}"
    done
  done
done
python aggregate_folds.py
