# Diabetic Retinopathy Severity Grading + Lesion Localization

A deep learning project to classify diabetic retinopathy (DR) severity from
retinal fundus photographs, with interpretability (Grad-CAM) used to verify
the model is looking at real lesions rather than confounding image
properties — framed as a clinical triage system (refer / no-refer), not just
a 5-class classifier.

## Problem

Diabetic retinopathy is a leading cause of preventable blindness, graded on
a standard 5-point scale from retinal fundus photographs:

| Grade | Name | Description |
|---|---|---|
| 0 | No DR | Healthy retina |
| 1 | Mild | A few microaneurysms |
| 2 | Moderate | More microaneurysms + hemorrhages |
| 3 | Severe | Extensive hemorrhages, venous beading |
| 4 | Proliferative | Abnormal new blood vessel growth — sight-threatening |

In real deployment, this model would support screening in settings with too
few ophthalmologists — flagging patients who need urgent specialist referral
(grades 3-4) rather than just predicting a grade in isolation.

## Dataset

- **Source:** APTOS 2019 Blindness Detection
- **Splits:** pre-defined train (2930 images) / valid (366) / test (366),
  each with its own `id_code` → `diagnosis` (0-4) CSV
- **No patient ID field available** in this public version — a strict
  patient-level leakage check isn't possible with the given metadata. Noted
  as a documented limitation; duplicate-image detection (below) is used as
  the practical substitute.

## Repo Structure

```
project/
├── notebooks/
│   └── 01_eda.ipynb                  # this phase — see below
├── src/
│   └── data_utils.py                 # reusable functions, imported by notebooks
├── archive/                          # dataset (train/valid/test images + CSVs)
├── outputs/
│   └── figures/                      # saved EDA charts
└── README.md
```

**Convention going forward:** any function used in more than one place lives
in `src/`, imported into notebooks — not re-pasted. One-off investigative
code (answers a single question once) stays inline in the notebook it
belongs to.

---

## EDA Summary (`01_eda.ipynb`) — Complete

### 1. Data integrity

| Check | Result |
|---|---|
| Missing / corrupted files | 0 / 0 — full integrity confirmed |
| Duplicate images within train | 79 pairs found and de-duplicated |
| Train↔Valid leaked images | 22 found and removed from train |
| Train↔Test leaked images | 22 found and removed from train |

**Why this mattered:** ~6% of valid/test had a byte-identical duplicate
sitting in train. Left unfixed, validation/test metrics would have been
artificially inflated by memorization rather than genuine generalization.
Leaked images were removed from `train` only, preserving valid/test as
untouched evaluation sets. Post-cleaning training set: **2,930 → ~2,886**
images (all 5 classes retained, no class was disproportionately gutted).

### 2. Class imbalance

| Grade | Count (post-cleaning) | % of data | Imbalance ratio vs. majority |
|---|---|---|---|
| 0 | 1432 | ~53% | 1.00x |
| 1 | 291 | ~11% | 4.92x |
| 2 | 784 | ~29% | 1.83x |
| 3 | 152 | ~5.6% | **9.42x** |
| 4 | 227 | ~8.4% | 6.31x |

**Key tension:** grades 3 and 4 (severe / proliferative — the clinically
most important cases to catch) are also the rarest. Accuracy alone will be
a misleading metric. **Decision (for the modeling phase): do not drop data
to balance classes** — rare-class images are too valuable. Instead, handle
imbalance at the training level (class weights, focal loss, or targeted
augmentation of minority classes), and evaluate using **Quadratic Weighted
Kappa (QWK)** plus per-class recall, not plain accuracy.

### 3. Image properties

- **Resolution:** highly variable — 819×614 up to 4288×2848 (mean ~1970×1505).
  Confirms resizing to a fixed input size (e.g. 224×224 or 300×300) is
  mandatory; note this discards more detail from the largest images than
  the smallest.
- **Channels:** consistently RGB (3 channels) across the board — no
  grayscale/alpha surprises to handle.
- **Aspect ratio:** ranges from ~1.0 (square) to ~1.51 (rectangular). A
  naive resize-to-square will distort the more rectangular images —
  **pad-to-square before resizing is safer than center-crop**, since
  cropping risks cutting off real retinal tissue at the edges.
- **Border/framing:** inconsistent across images — some show the full
  circular retina with black padding, others are cropped tight into the
  tissue. Needs a consistent cropping strategy before resizing.

### 4. Blur / sharpness — real, flagged finding

Using Laplacian variance (edge-sharpness proxy) across train/valid/test:

| Grade | Mean blur score (train) |
|---|---|
| 0 | ~37 |
| 1 | ~14 |
| 2 | ~13 |
| 3 | ~14 |
| 4 | ~15 |

**Grade 0 (healthy) images are consistently ~2.5x sharper than every
diseased class, across all three splits.** This is counter-intuitive
(diseased retinas have more fine detail/lesions, so you'd expect them to
score *higher*, not lower). Two possible explanations, not yet
distinguished:
1. Genuine signal — diabetic cataracts (a separate diabetes complication)
   cause lens haze that correlates with disease severity.
2. Confound — more severe cases may have been captured with different
   equipment/protocol.

**Decision: do NOT filter or drop images based on blur** — it may carry
real diagnostic signal, and filtering risks disproportionately harming the
already-rare grade 3 class. **Action for later:** explicitly check during
Grad-CAM whether the model is keying off real lesions vs. overall image
haze as a shortcut.

### 5. Color-channel hypothesis — tested, ruled out

Initial visual impression: grade 0 looked more yellow, diseased grades more
reddish. Quantitative per-channel (R/G/B) means across 300 samples did
**not** show a consistent trend — R:G ratio stayed ~1.8-2.1 across all
classes with no monotonic pattern by grade. Concluded this was
sample-size-driven visual bias, not a true confound. Documented as a
negative result — a deliberately tested and ruled-out hypothesis, not just
an unchecked hunch.

---

## Decisions Carried Into Preprocessing

1. **Crop** to the circular retina region, standardize black border.
2. **Pad-to-square before resize** (not center-crop) to avoid distorting
   rectangular images or cutting off tissue.
3. **Resize** to a fixed size (224×224 or 300×300 — final choice depends on
   compute budget, see below).
4. **Contrast enhancement (CLAHE)** worth testing — lesions are low-contrast
   on some images.
5. **Do not discard images** to fix class imbalance — handle via class
   weights / focal loss / minority-class augmentation at training time.
6. **Do not filter by blur** — keep as-is, revisit only via Grad-CAM later.
7. **Rotation/flip augmentation** is safe (retinas have no fixed
   orientation); be cautious with color-based augmentation since color may
   carry diagnostic signal.
8. **Evaluation metric:** QWK (primary) + per-class recall + confusion
   matrix — not plain accuracy.

## Environment Notes

- Training locally on an RTX 3050 (4GB VRAM laptop GPU).
- Mixed precision (`mixed_float16`) enabled — roughly halves memory use,
  supported natively on this GPU.
- Backbone choice: start with EfficientNetB0 or MobileNetV3-Large (fits
  comfortably in 4GB); avoid B3+ until a compute upgrade or Colab Pro.
- Batch size: start at 16, drop to 8 if OOM.
- `tf.data` pipeline (`.cache().shuffle().batch().prefetch()`) required —
  not optional — to keep the GPU fed without CPU-side bottlenecks.

## Project Roadmap

- [x] **Phase 1 — EDA** (`01_eda.ipynb`): integrity, imbalance, image
      properties, blur, color — complete, see summary above
- [ ] **Phase 2 — Preprocessing pipeline** (`02_preprocessing_pipeline.ipynb`):
      cropping, resizing, `tf.data` pipeline, augmentation, class-weight
      computation
- [ ] **Phase 3 — Baseline training** (`03_baseline_training.ipynb`):
      transfer learning with EfficientNetB0/MobileNetV3, frozen backbone →
      fine-tune
- [ ] **Phase 4 — Imbalance experiments** (`04_class_imbalance_experiments.ipynb`):
      compare class weights vs. focal loss vs. augmentation-based approaches
- [ ] **Phase 5 — Evaluation** (`05_evaluation_analysis.ipynb`): QWK,
      confusion matrix, per-class recall, referral-threshold analysis
      (binary refer/no-refer framing)
- [ ] **Phase 6 — Interpretability** (`06_gradcam_interpretability.ipynb`):
      Grad-CAM overlays; explicitly check whether predictions correlate with
      real lesions vs. overall image blur/haze
- [ ] **Phase 7 — Deployment** (`07_final_model_export.ipynb` + Streamlit/HF
      Space): upload image → grade + Grad-CAM overlay + referral flag

## Known Limitations

- No patient-ID metadata available — cross-session/patient leakage beyond
  exact-duplicate images cannot be fully verified.
- Exact-duplicate detection (MD5 hashing) does not catch near-duplicates
  (same eye re-photographed with different compression/crop/rotation).
- Resize step discards more relative detail from the highest-resolution
  images than the lowest.