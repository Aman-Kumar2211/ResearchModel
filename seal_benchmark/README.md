# Seal / tape tamper classification: reproducible robustness benchmark

Fine-tunes two lightweight CNNs (MobileNetV3-Small, EfficientNet-B0) on clean photos of
taped box seams, then tests them on real occluded / low-light phone photos and on
synthetic corruptions. Same dataset + same manifest + same seeds + same package
versions => same results.

## 1. Setup
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip freeze > requirements.lock.txt      # keep this with your results
```
Pretrained weights download once from the Hugging Face hub (internet needed on first run).
They are cached afterwards; set `HF_HUB_OFFLINE=1` to be sure nothing changes later.

## 2. Dataset layout and file naming
Put every photo in `dataset/images/` (subfolders are allowed). Name each file:

    {box}_{label}_{tamper}_{phone}_{lighting}_{angle}_{occlusion}_{idx}.jpg
    I08_tampered_cut_B_daylight_straight_none_01.jpg
    I08_intact_none_B_lowflash_straight_finger2_03.jpg

| token | allowed values |
|---|---|
| box | letter + digits. `I01`..`I10` = identical boxes, `D01`.. = different boxes |
| label | `intact`, `tampered` |
| tamper | `none` (intact only), `peel`, `cut`, `restuck`, `disguise` |
| phone | one capital letter: `A`, `B`, `C`, `D`, ... |
| lighting | `daylight`, `indoor`, `lowlight`, `lowflash`, `glare` |
| angle | `straight`, `offaxis` |
| occlusion | `none`, `low` (light finger), `finger2` (moderate), `high` (heavy finger), `sticker`, `shadow`, `partialframe` |
| idx | digits, to keep names unique |

Older spellings (`peeled`, `partial`, `disguised`, `finger1`, `finger3`) in existing filenames are still accepted and converted to the new names in `manifest.csv`.

A photo is **clean** if lighting is `daylight`/`indoor` and occlusion is `none`; otherwise
it is a **robustness** photo. Training and validation use clean photos only.

Copy `splits.example.json` to `splits.json` and edit it: it maps each **box** to
`train`, `val` or `test`. Splitting is by box, never by photo, so near-duplicate photos of one
box cannot appear on both sides.

### 2b. From camera photos to this layout
1. Copy the ORIGINAL photos from each phone (cable, Drive "original quality", or send as a Document). Photos sent as
   normal WhatsApp images are re-compressed and lose their rotation data.
2. Sort them into folders: `raw/<box>/<state>/<phone>/<lighting>_<angle>_<occlusion>/` (any camera file names inside), e.g.
   `raw/I03/tampered/B/indoor_straight_finger3/`.
3. Copy `tampers.example.csv` to `tampers.csv` (box -> tamper type), then:
   `python organize_photos.py --raw raw --tampers tampers.csv --dry-run` to preview, then without `--dry-run` to
   copy the files into `dataset/images/` with the correct names. `rename_log.csv` records every original -> new name.

## 3. Run
```
python make_manifest.py --images dataset/images --splits splits.json --holdout-phones D --out manifest.csv
python corruptions.py --demo dataset/images/<any photo>.jpg      # eyeball the corruptions
python train.py    --model mobilenetv3_small_100 --seed 0
python evaluate.py --run runs/mobilenetv3_small_100_seed0
python aggregate.py
```
or everything at once: `./run_all.sh` (2 models x 3 seeds).

Smoke test before real data (random weights, no download, meaningless accuracy):
`python train.py --model mobilenetv3_small_100 --seed 0 --no-pretrained --stage1-epochs 1 --stage2-epochs 1 --out runs_smoke`
then `python evaluate.py --run runs_smoke/mobilenetv3_small_100_seed0 --no-pretrained --speed-runs 5`.

Check the dataset has not changed since the manifest was made:
`python make_manifest.py --images dataset/images --verify manifest.csv`

## 4. What is fixed (so results reproduce)
- **Split and labels:** `manifest.csv` (SHA-256 of every photo). Each run stores the manifest hash;
  `evaluate.py` refuses to run if the manifest differs from the one used for training.
- **Preprocessing:** EXIF rotation applied, resized to 224x224 (bicubic, aspect squashed), ImageNet normalization from the model's own config.
- **Seeds:** `--seed` controls head initialisation, shuffling and augmentation. Deterministic kernels requested; DataLoader uses 0 workers; device defaults to CPU.
- **Corruptions:** generated per image from `CORRUPTION_SEED` in `config.py`, independent of the training seed, so every model and seed sees identical corrupted test images.
- **Training recipe** (defaults in `train.py`, saved in each run's `config.json`): stage 1 trains only the new head for 4 epochs (lr 1e-3, backbone frozen and in eval mode); stage 2 fine-tunes everything for up to 12 epochs (backbone 1e-4, head 1e-3, AdamW, weight decay 1e-4, cosine schedule), early stopping on validation loss (patience 3), class-weighted cross-entropy. Augmentation is light only: random resized crop (0.85-1.0), horizontal flip, rotation up to 10 degrees.
- **Environment:** Python / torch / timm / Pillow versions and git commit are written into each `config.json`.

## 5. What can still differ
- Different machines, CPU vs GPU, or different library versions can change results slightly. Compare against your own `requirements.lock.txt`.
- JPEG corruption depends on the Pillow/libjpeg build.
- Latency in `speed.json` is a timing measurement, not a reproducible number. Report mean +/- std and the hardware.
- Pretrained weights come from the hub; the exact weight tag is recorded in `config.json`.

## 6. Outputs and where they go in the paper
| file | use |
|---|---|
| `runs/<model>_seed<k>/history.csv, config.json, best.pt` | training log, full settings, weights |
| `runs/<...>/results/predictions.csv` | every prediction with P(tampered); recompute any metric or threshold |
| `runs/<...>/results/summary.csv` | per-run metrics for all breakdowns |
| `results/summary_mean_std.csv` | mean +/- std over seeds (Tables 1, 3, 4) |
| `results/efficiency.csv` | params, size, CPU latency (Table 2) |
| `results/accuracy_vs_severity_*.png` | Figure 1 |

`analysis` values in the summaries: `real_set` (group|condition), `real_occlusion`, `real_lighting`,
`real_phone`, `real_phone_seen` (unseen-phone test), `real_tamper_type` (recall per tamper type on clean photos),
and `synthetic` (group|corruption|severity). Positive class = tampered; `fnr` is the share of tampered photos called intact.

## 7. Honest limits to state in the paper
With ~6 training boxes the effective sample size is the number of boxes, not photos. Report mean +/- std
over seeds, say the study is a feasibility test, and do not read small differences between models as significant.
Optional stronger design: make several `splits.json` files that rotate which identical boxes are held out,
run the pipeline for each, and average.

## 8. Grouped k-fold (main results once the single-split pilot works)
Run the single split first to check photos, names and scripts. Then:
```
./run_kfold.sh          # make_folds -> train + evaluate every (fold, model, seed) -> aggregate_folds
```
- `make_folds.py` turns the identical boxes into k folds (default 5). In each fold 2 identical boxes are
  test, 1 is validation (borrowed from the next fold, chosen so every tamper type stays in training when
  possible) and the rest train. The different boxes are test-only in every fold. Each identical box is tested exactly once.
- `aggregate_folds.py` averages seeds inside each fold, then reports mean +/- std **across folds**
  (`results_kfold/summary_mean_std_over_folds.csv`, `summary_per_fold.csv`, figures). `n_folds` shows
  how many folds contributed to each row (a tamper type or occlusion missing from a fold's test boxes is not counted).
- One seed per fold is enough; add seeds only if time allows. `TRAIN_ARGS` in `run_kfold.sh` is where to change epochs / patience
  after checking `history.csv` from the pilot (for example `--stage2-epochs 30 --patience 5`).

**What k-fold changes about photo collection**
1. Every identical box is a test box in one fold, so take the **robustness photos (low light, finger levels, sticker,
   shadow, glare, partial frame) for all identical boxes**, not just three. Otherwise a fold's robustness results are empty.
2. The unseen-phone test: with `HOLDOUT_PHONES="D"`, phone D photos are used only when their box is a test box;
   they are dropped (`split=unused`) for train and val boxes so the same physical box never appears on both sides.
   Photograph all identical boxes with phone D if you want it in every fold.
3. The different boxes keep the same role in every fold, so their spread across folds reflects training differences only.
