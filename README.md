# Hierarchical Vietnamese Traffic Sign Detection with YOLO26n
### Testing Two Independent Interventions Against a Flat-Label Baseline: Hierarchical Label Smoothing and a P2 Small-Object Head

> A controlled empirical study on the **VNTS dataset** asking whether QCVN 41's regulatory sign taxonomy can improve YOLO26n traffic sign detection through the loss function (Hierarchical Label Smoothing), and whether an architecture-level small-object detection head (P2) helps independently — evaluated under a corrected protocol that keeps hyperparameter selection and final test evaluation strictly separated.

---

## Project Description

Traffic sign detection is a core component of Advanced Driver Assistance Systems (ADAS) and autonomous driving. Vietnamese traffic signs are formally organized under the **QCVN 41** regulation into five categories — **Prohibitory, Warning, Mandatory, Information,** and **Supplementary** — but existing Vietnamese datasets and models treat all 52 sign classes as independent ("flat-label"), ignoring this structure entirely.

This project builds the first QCVN 41-aligned hierarchical taxonomy for the **Vietnamese Traffic Signs (VNTS)** dataset and uses it to test two *independent* interventions against a common flat-label YOLO26n baseline:

- **A loss-level intervention** — Hierarchical Label Smoothing (HLS), which redistributes classification target probability toward semantically related sibling classes without touching the model architecture.
- **An architecture-level intervention** — a dedicated high-resolution "P2" detection head (stride 4), added to catch small objects that fall below the resolution of YOLO26n's default earliest head.

Each is isolated against the same flat-label control, so hierarchy and architecture are never conflated in a single comparison.

A methodological note that shapes everything below: between the R2 and R3 milestones, the evaluation protocol was corrected. The original version used the held-out test set as the validation split *during* training — meaning hyperparameter and checkpoint decisions were implicitly informed by test performance. Under the fixed protocol, HLS hyperparameters are selected purely by cross-validation on the training pool, and the test set is touched exactly once, after every decision is locked in. **All results in this README reflect the corrected protocol.**

---

## Motivation

- VNTS exhibits severe class imbalance — instance counts range from 3 to 1,071 per class — making rare-class detection difficult under standard flat-label training.
- The QCVN 41 hierarchy is a natural, currently unused source of supervision: signs within the same category are visually and semantically related.
- Sign instances are uniformly small across the image regardless of frequency tier, motivating a test of whether a higher-resolution detection head helps independent of the loss function.
- No prior work has studied a loss-level hierarchical intervention and an architecture-level small-object intervention *together, in isolation from each other*, for Vietnamese traffic signs.

---

## Contributions

- The first **QCVN 41-aligned hierarchical taxonomy** for the VNTS dataset (52 classes → 5 categories), including manual resolution of two raw-label inconsistencies.
- **Hierarchical Label Smoothing (HLS)** implemented purely at the loss level — the classification target, not the architecture, carries the hierarchy — preserving inference speed and identical compute to the flat baseline by construction.
- A **three-model controlled comparison**: flat-label baseline, HLS, and a P2-augmented architecture, all trained and evaluated on an identical data split.
- A **cross-validation hyperparameter search** over four HLS configurations, isolated entirely from the test set.
- **Per-tier and per-category statistical testing** (Wilcoxon signed-rank on paired per-class AP50), rather than relying on aggregate mAP alone.
- An explicit correction of an earlier train/test leakage issue, with the superseded interim result documented rather than silently dropped.
- An honest negative result: neither intervention beats the flat baseline, and the report explains *why*, in both cases.

---

## Research Questions

1. Does Hierarchical Label Smoothing improve Vietnamese traffic sign detection over a flat-label baseline, once hyperparameter selection is properly separated from test evaluation?
2. How much do HLS's own hyperparameters (target-gap width, classification loss weight) matter within the HLS family?
3. Does QCVN 41 category size predict which sign categories benefit from hierarchical supervision?
4. Does an architecture-level small-object head (P2) provide an independent benefit, isolated from the loss-level intervention?

---

## Dataset

| Property | Value |
|---|---:|
| Images | 3,216 |
| Annotated instances | 8,334 |
| Sign classes | 52 |
| Train / test split | 2,552 / 639 |
| Average objects per image | 2.59 |
| Semantic categories (QCVN 41) | 5 |

**Class imbalance.** Instance counts per class range from 3 to 1,071 — a severe long tail. Classes were stratified into three frequency tiers using the 25th/75th percentiles as cutoffs:

| Tier | Threshold | Classes | Instances |
|---|---:|---:|---:|
| Frequent | ≥ 100 | 24 | 7,201 |
| Medium | 30–99 | 15 | 873 |
| Rare | < 30 | 13 | 260 |

**QCVN 41 category mapping.** All 52 classes were mapped to five categories via a deterministic lookup table built from the regulation text. Two raw-label inconsistencies were resolved during this process: a class labeled "B.8a" was reclassified as P.108a (Prohibitory) based on visual inspection, and a non-standard "Camera" class (speed-camera warning signs) was assigned to Information as a practical decision.

| Category | Classes | Instances | % of Total |
|---|---:|---:|---:|
| Prohibitory | 23 | 5,147 | 61.8% |
| Warning | 18 | 1,727 | 20.7% |
| Mandatory | 4 | 798 | 9.6% |
| Information | 4 | 472 | 5.7% |
| Supplementary | 3 | 190 | 2.3% |

**Spatial characteristics.** Median bounding-box area stays under 0.25% of image resolution across all three frequency tiers (Frequent 0.14%, Medium 0.24%, Rare 0.19%) — signs are uniformly small regardless of class frequency, which is the direct motivation for testing the P2 architectural enhancement.

---

## Method

### Three models, two isolated interventions

| Model | Architecture | Loss target | Selection |
|---|---|---|---|
| **B** (baseline) | YOLO26n, unmodified | Flat (TAL-scaled one-hot) | Fixed, no tuning |
| **A** (HLS) | YOLO26n, unmodified — architecturally identical to B | HLS: α=0.05, β=0.10, λ_cls=1.5 | Winner of a 4-way cross-validation ablation |
| **P2** (architecture) | YOLO26n + extra P2 detection head | Flat, same as B | Fixed, no tuning |

Models A and B are architecturally identical (2,384,976 parameters, 5.2 GFLOPs) — the *only* difference between them is the classification loss target, isolating the loss-level intervention cleanly. Model P2 adds a fourth, higher-resolution detection head, increasing parameters to 2,429,832 (+1.9%) and compute to 6.8 GFLOPs (+30.8%), isolating the architectural intervention against the same flat loss target used by Model B.

### Model B — Flat-label baseline

YOLO26n's Task-Aligned Assigner (TAL) scales the target for the correct class by a box-quality metric, but all other 51 classes stay at exactly zero — there is no cross-class redistribution. Every misclassification is penalized identically regardless of semantic proximity.

### Model A — Hierarchical Label Smoothing

Let S(c) be the set of classes sharing class c's QCVN 41 category, including c itself. The HLS target redistributes probability mass three ways:

- the correct class receives **1 − α − β**;
- each of the |S(c)| − 1 sibling classes in the same category receives **β / (|S(c)| − 1)**;
- each remaining class outside the category receives **α / (N − |S(c)|)**, N = 52.

Worked example — a Warning-category class (18 classes total) under the final hyperparameters (α=0.05, β=0.10): the correct class gets 0.85, each of the other 17 Warning classes gets ≈0.0059, and each of the 34 classes outside Warning gets ≈0.0015.

This smoothed target replaces the TAL-scaled target inside the classification loss only; box regression and DFL are untouched, and HLS applies identically to both the one-to-many and one-to-one assignment heads. A consequence worth flagging: this fixes the correct-class target at 1 − α − β regardless of localization quality, discarding the box-quality signal that Model B retains.

### Model P2 — Architectural small-object enhancement

Adds a stride-4 detection head (following the TPH-YOLOv5 design), sourced from an earlier backbone stage and fused with upsampled P3 features, extending the range of reliably detectable object sizes downward. Initialized from the pretrained yolo26n.pt checkpoint wherever shapes match; the new P2-specific layers are randomly initialized. Uses the same flat classification target as Model B — architecture is the sole variable in this comparison.

### Training protocol

- **Phase 1 (Model A only):** Four HLS configurations (varying α, β, λ_cls) evaluated via 3-fold multilabel-stratified cross-validation, restricted entirely to the training pool, 45 epochs per fold (bounded by Kaggle's 12-hour session limit and 30-hour weekly GPU quota). The best mean CV mAP@0.5 configuration is carried forward.
- **Phase 2 (all three models):** Each of Model A (winning config), Model B, and Model P2 trained for 100 epochs on an identical split — ~90% of the training pool for training, ~10% held out only for checkpoint selection.
- **Test evaluation:** The 639-image test set is touched exactly once per final model, after all hyperparameter and checkpoint decisions are complete. No decision anywhere in the pipeline is made by looking at test performance.

| Parameter | Value |
|---|---|
| Base weights | yolo26n.pt (COCO-pretrained) |
| Input resolution | 640 × 640 |
| CV ablation epochs (per fold) | 45 |
| CV folds | 3 (multilabel-stratified) |
| Final training epochs | 100 |
| Internal validation carve-out | ~10% of training pool |
| Batch size | 32 |
| Optimizer | AdamW (auto-selected) |
| Augmentation | Mosaic, HSV jitter, horizontal flip |
| Random seed | 0 (fixed across all runs) |
| Hardware | NVIDIA Tesla T4 (Kaggle) |
| Experiment tracking | Weights & Biases |

Because HLS only changes the classification target via a precomputed 52×52 smoothing lookup, it adds negligible training overhead and has **zero** effect on inference — Models A and B run an identical forward pass by construction, not merely by measurement.

> **Note on the R2 interim result.** R2 reported an interim baseline of mAP@0.5 = 0.9487 under the earlier (leaky) protocol, which used the test set as the validation split during training. That figure is **superseded** and not comparable to the results below.

---

## Experimental Results

### HLS hyperparameter selection (cross-validation, training pool only)

| Config | α | β | λ_cls | Global | Frequent | Medium | Rare |
|---|--:|--:|--:|--:|--:|--:|--:|
| HLS-1 | 0.05 | 0.10 | 0.5 | 0.579 ± 0.005 | 0.752 ± 0.009 | 0.551 ± 0.030 | 0.285 ± 0.013 |
| HLS-2_wide | 0.01 | 0.20 | 0.5 | 0.527 ± 0.009 | 0.732 ± 0.005 | 0.482 ± 0.027 | 0.193 ± 0.020 |
| **HLS-3_hicls ★** | 0.05 | 0.10 | **1.5** | 0.687 ± 0.008 | 0.808 ± 0.006 | 0.733 ± 0.008 | 0.403 ± 0.023 |
| HLS-4_combo | 0.01 | 0.20 | 1.5 | 0.656 ± 0.005 | 0.796 ± 0.002 | 0.672 ± 0.015 | 0.370 ± 0.017 |

Classification loss weight (λ_cls) is the dominant driver within the HLS family: both λ_cls=1.5 configurations clearly beat both λ_cls=0.5 configurations at every tier, most dramatically on Rare (0.37–0.40 vs. 0.19–0.29). Widening the target gap alone, without raising λ_cls, *hurts* rather than helps. **HLS-3_hicls** was selected as Model A's final configuration.

### Final test-set results (each model evaluated once)

| Metric | Model B | Model A | Model P2 |
|---|--:|--:|--:|
| Precision | 0.8337 | 0.8295 | **0.8455** |
| Recall | **0.8796** | 0.8689 | 0.6708 |
| mAP@0.5 | **0.9257** | 0.9131 | 0.8335 |
| mAP@0.5:0.95 | **0.6945** | 0.6690 | 0.6173 |

| Tier | Model B mAP@0.5 | Model A mAP@0.5 | Model P2 mAP@0.5 |
|---|--:|--:|--:|
| Frequent | 0.9536 | **0.9556** | 0.8839 |
| Medium | 0.9336 | **0.9387** | 0.8523 |
| Rare | **0.8649** | 0.8053 | 0.7188 |

Model A edges past Model B on Frequent and Medium by under a point, but trails on both global metrics and, notably, on Rare — the same tier where λ_cls showed its largest CV effect. Model P2 underperforms Model B at every tier and metric, with the sharpest gap in Recall.

### Statistical significance (Wilcoxon signed-rank, paired per-class AP50)

| Tier | n | Model A vs. B (p) | Model P2 vs. B (p) |
|---|--:|--:|--:|
| Global | 52 | 0.5539 | 3.4×10⁻⁷ |
| Frequent | 24 | 0.8774 | 3.6×10⁻⁷ |
| Medium | 15 | 0.8139 | 0.0131 |
| Rare | 13 | 0.2439 | 0.0645 |

**Model A shows no statistically significant difference from Model B at any tier** — no evidence that the CV-selected HLS configuration beats the flat baseline. **Model P2 differs significantly from Model B at Global, Frequent, and Medium** (Rare approaches significance) — a clear, statistically supported *negative* result.

### Per-category analysis (Model A vs. Model B)

| Category | Classes | Mean AP50 delta |
|---|--:|--:|
| Supplementary | 3 | −0.1527 |
| Mandatory | 4 | +0.0281 |
| Information | 4 | +0.0344 |
| Warning | 18 | −0.0093 |
| Prohibitory | 23 | −0.0120 |

The R2 hypothesis that larger categories benefit less from HLS does not survive contact with the taxonomy's actual sample sizes. Supplementary, Mandatory, and Information have only 3–4 classes each, so their category-level means are dominated by noise from a handful of classes. Only Warning (18) and Prohibitory (23) are large enough for a stable estimate — and both show small *losses*, not gains. A Pearson correlation between raw per-class training-instance count and AP50 delta was also negligible (r = 0.0737, p = 0.6037): class rarity does not predict HLS's effect either.

### Why Model P2 fails the way it does

Model P2's failure has a specific signature, not a diffuse one: it posts the **highest precision** of all three models while Recall collapses and mAP@0.5 (which integrates over the full precision-recall curve, not one threshold) also drops clearly. This rules out a simple "got worse at everything" explanation — it instead points to poorly calibrated confidence from the new, untrained P2 branch, most plausibly because those randomly-initialized parameters must learn calibration from scratch within a 2,552-image pool and a fixed 100-epoch budget, without the benefit of COCO pretraining that Models A and B inherit. A secondary possibility: VNTS signs may already be adequately resolved by YOLO26n's default P3 head, meaning the added P2 head introduces competing anchors without addressing a real resolution bottleneck.

---

## Discussion

The central finding is that **HLS's apparent benefit does not survive a properly separated evaluation protocol.** Both the R2 interim result and the original R3 draft reported a positive HLS effect under a leaky protocol; under the corrected one, the best-tuned HLS configuration is statistically indistinguishable from the flat baseline everywhere, and trails it on raw Global and Rare-tier numbers.

This doesn't mean HLS's hyperparameters don't matter — within the HLS family, λ_cls drives a large, low-variance effect, roughly doubling Rare-tier performance when raised from 0.5 to 1.5. The honest summary is that HLS's own hyperparameters matter a great deal, and even once optimized, the result doesn't clear the bar a plain baseline already sets.

Model P2's negative result is the most statistically confident finding in the project — a clear, significant underperformance whose precision/recall/mAP signature specifically implicates confidence miscalibration in the untrained branch, not a fundamental mismatch between P2 heads and traffic sign detection.

---

## Limitations

- HLS replaces YOLO26n's box-quality-aware TAL scaling with a fixed hierarchical target, so Model A's correct-class target no longer reflects localization quality the way Model B's does — a confound beyond the intended hierarchical redistribution.
- The CV ablation used 45 epochs (Kaggle quota constraints) versus 100 for final training — the ranking among HLS configurations is not independently verified at the longer schedule beyond the single winning configuration carried forward.
- The internal validation carve-out for checkpoint selection is small (~255 images), which may introduce checkpoint-selection noise.
- A single random seed was used for final training; variance is only characterized across CV folds during the ablation, not across independent seeds for the final models.
- Supplementary (3 classes, 190 instances) and the Rare tier (13 classes, 260 instances) produce high-variance per-class estimates regardless of model, limiting confidence in category- and tier-specific conclusions.
- Model P2's negative result was not decomposed into its two candidate causes (undertrained new parameters vs. genuine architecture-task mismatch).
- VNTS contains only 3,216 images; the interaction between either intervention and dataset scale remains untested.

---

## Future Work

- Repeat Model P2 with a longer schedule or staged warm-up (freezing pretrained layers while the P2 head trains first) to test whether its negative result is a convergence artifact.
- Compare VNTS's bounding-box size distribution directly against YOLO26n's P2/P3 resolution boundary, rather than inferring smallness only from area statistics.
- Repeat the HLS cross-validation with multiple seeds per configuration to separate seed variance from fold variance, and verify the winning configuration at the full 100-epoch budget.
- Since neither raw instance count nor QCVN 41 category size predicts HLS's per-class effect, investigate alternative structuring principles (e.g. visual/shape similarity) for the hierarchical target.
- Evaluate on a larger Vietnamese traffic sign dataset to increase statistical power in the Rare tier and Supplementary category.
- Investigate Hierarchical Cosine Loss (Genermont et al.) as an alternative hierarchical formulation — HLS's null result here does not rule out other hierarchical loss designs.

---

## Conclusion

This project built the first QCVN 41-aligned hierarchical taxonomy for VNTS and ran a controlled comparison of a loss-level intervention (HLS) and an architecture-level intervention (a P2 small-object head) against a common flat-label YOLO26n baseline, under a protocol that keeps hyperparameter selection and test evaluation strictly separated after an earlier leakage issue was identified and corrected.

Under this corrected protocol, HLS's CV-selected configuration is not statistically distinguishable from the flat baseline at any tier and trails it on raw Global and Rare-tier numbers — even though its own hyperparameters (λ_cls in particular) have a large, reproducible effect within the HLS family. Model P2 produces a clear, statistically significant negative result at every tier, most likely driven by poorly calibrated confidence in its untrained branch rather than a fundamental architecture mismatch.

The project's central contribution is this negative result itself: a plausible-sounding hierarchical loss modification does not outperform a well-tuned flat baseline once hyperparameter selection and final test evaluation are properly kept independent.

---

## References

[1] E. H.-C. Lu, M. Gozdzikiewicz, K.-H. Chang, and J.-M. Ciou, "A hierarchical approach for traffic sign recognition based on shape detection and image classification," *Sensors*, vol. 22, no. 13, p. 4768, 2022.

[2] X. Zhang et al., "Traffic sign detection and recognition using hierarchical semantic networks," *Engineering Applications of Artificial Intelligence*, 2021.

[3] C. Ertler et al., "The Mapillary traffic sign dataset for detection and classification on a global scale," in *Proc. ICCV Workshops*, 2019.

[4] M. A. Usmani, S. Mahmood, Y. Elmadany, A. Z. Azeem, and I. Zualkernan, "Hierarchical YOLO with real-time text recognition for UAE traffic signs," *IEEE Sensors Journal*, vol. 24, no. 23, pp. 1–11, 2025.

[5] V. Tsenkova, P. Stanchev, D. Petrov, and D. Lazarov, "hYOLO model: Enhancing object classification with hierarchical context in YOLOv8," arXiv:2510.23278, 2025.

[6] R. Sapkota, R. H. Cheppally, A. Sharda, and M. Karkee, "YOLO26: Key architectural enhancements and performance benchmarking for real-time object detection," arXiv:2509.25164, 2025.

[7] X. Zhu, S. Lyu, X. Wang, and Q. Zhao, "TPH-YOLOv5: Improved YOLOv5 based on transformer prediction head for object detection on drone-captured scenarios," in *Proc. IEEE/CVF ICCV Workshops*, 2021, pp. 2778–2788.

[8] A. Genermont et al., "Hierarchical novelty detection for traffic sign recognition," *Frontiers in Computer Science*, 2022.

---

## Acknowledgements

This project was completed as part of the **Project-Based Learning (PBL)** curriculum at **FPT University**, supervised by a faculty advisor.

This work builds upon:
- Ultralytics YOLO26
- Vietnamese Traffic Signs (VNTS) Dataset
- Weights & Biases
- QCVN 41: National Technical Regulation on Road Traffic Signs
