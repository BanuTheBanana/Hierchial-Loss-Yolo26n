# Hierarchical Vietnamese Traffic Sign Detection using YOLO26n

> Improving Vietnamese traffic sign detection by incorporating the QCVN 41 traffic sign hierarchy into YOLO26n through Hierarchical Label Smoothing (HLS).

<!-- ===================================================================================== -->
<!-- TODO: Replace with a banner image or project logo                                     -->
<!-- Recommended size: ~1200 x 400                                                         -->
<!-- ===================================================================================== -->

<p align="center">
  <img src="images/banner.png" width="90%">
</p>

---

## Project Description

This repository contains the code, experiments, and resources for the project **"Hierarchical Vietnamese Traffic Sign Detection using YOLO26n and Hierarchical Label Smoothing."** The objective of this work is to investigate whether incorporating the semantic hierarchy defined by the Vietnamese **QCVN 41** traffic sign regulation can improve object detection performance without modifying the YOLO26n architecture. Instead of treating all 52 traffic sign classes as independent labels, the proposed **Hierarchical Label Smoothing (HLS)** method redistributes the classification target among semantically related sibling classes while preserving the original inference pipeline.

The project is evaluated on the **Vietnamese Traffic Signs (VNTS)** dataset through a controlled comparison against a flat-label baseline, followed by multiple hyperparameter ablation studies. Besides evaluating detection performance, the project also investigates *when* hierarchical supervision is beneficial, *why* it succeeds or fails under different configurations, and the limitations of applying regulatory hierarchies to object detection.

Key contributions include:

- Hierarchical annotation of the VNTS dataset based on the QCVN 41 taxonomy.
- Implementation of Hierarchical Label Smoothing for YOLO26n.
- No architectural modifications and zero additional inference cost.
- Comprehensive hyperparameter ablation of HLS configurations.
- Statistical significance analysis using Wilcoxon signed-rank tests.
- Analysis of how semantic category size influences hierarchical supervision.
- Discussion of the practical limitations and failure cases of HLS.

<!-- ===================================================================================== -->
<!-- TODO: Add a qualitative comparison image.                                             -->
<!-- Example: Baseline vs HLS predictions on the same image.                               -->
<!-- ===================================================================================== -->

<p align="center">
  <img src="images/qualitative_results.png" width="90%">
</p>

---

## Experimental Highlights

| Configuration | α | β | λcls | mAP@0.5 |
|:--------------|--:|--:|------:|---------:|
| Baseline | — | — | 0.5 | **0.9487** |
| HLS | 0.05 | 0.10 | 0.5 | 0.8922 |
| HLS (Wide Gap) | 0.01 | 0.20 | 0.5 | 0.8735 |
| HLS (High Classification Weight) | 0.05 | 0.10 | 1.5 | 0.9219 |
| **HLS (Combined)** | **0.01** | **0.20** | **1.5** | **0.9583** |

**Main findings**

- The best HLS configuration achieved **95.83% mAP@0.5**, outperforming the flat-label baseline.
- Global performance improvement is statistically significant (Wilcoxon *p* = 0.0265).
- Increasing the classification loss weight is critical for hierarchical supervision.
- Category size, rather than class frequency, is the primary factor determining HLS effectiveness.
- Poorly calibrated HLS configurations may reduce detection performance, highlighting the importance of hyperparameter tuning.

<!-- ===================================================================================== -->
<!-- TODO: Replace with performance plots or ablation figure                               -->
<!-- ===================================================================================== -->

<p align="center">
  <img src="images/results.png" width="85%">
</p>

---

## Repository Structure

```text
Hierarchical-Loss-YOLO26n/
│
├── datasets/                 # VNTS dataset configuration
├── experiments/              # Experiment configurations
├── models/                   # Modified YOLO26n implementation
├── notebooks/                # EDA and visualization notebooks
├── scripts/                  # Training and evaluation scripts
├── results/                  # Saved checkpoints and evaluation outputs
├── paper/                    # Research paper
├── report/                   # Project report
├── images/                   # README figures
├── requirements.txt
└── README.md
```

---

## Installation

Clone this repository:

```bash
git clone https://github.com/<your_username>/Hierarchical-Loss-YOLO26n.git
cd Hierarchical-Loss-YOLO26n
```

Create a virtual environment (optional but recommended):

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

<!-- TODO:
If using a modified Ultralytics package, describe installation here.
Example:
pip install -e .
-->

---

## Dataset

The experiments are conducted on the **Vietnamese Traffic Signs (VNTS)** dataset obtained from Kaggle.

Dataset summary:

| Property | Value |
|----------|-------:|
| Images | 3,216 |
| Instances | 8,334 |
| Classes | 52 |
| Split | 2,552 train / 639 validation |

The original dataset is reorganized using the **QCVN 41** hierarchy into five semantic categories:

- Prohibitory
- Warning
- Mandatory
- Information
- Supplementary

<!-- TODO:
Provide the dataset download link.

If redistribution is not permitted, explain how users can obtain the dataset.
-->

Expected directory structure:

```text
dataset/
│
├── images/
│   ├── train/
│   └── val/
│
├── labels/
│   ├── train/
│   └── val/
│
└── vnts.yaml
```

---

## Usage

### Training the Baseline

```bash
python train_baseline.py
```

### Training with Hierarchical Label Smoothing

```bash
python train_hls.py
```

<!-- TODO:
Replace the commands above with your actual training commands.
If training uses Ultralytics CLI, include the exact command here.
-->

---

## Method Overview

The overall workflow is summarized below.

<!-- ===================================================================================== -->
<!-- TODO: Replace with your pipeline diagram                                              -->
<!-- ===================================================================================== -->

<p align="center">
  <img src="images/pipeline.png" width="95%">
</p>

Training pipeline:

```
VNTS Dataset
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Hierarchy Mapping (QCVN 41)
      │
      ▼
YOLO26n Training
      │
      ├──────────────┐
      ▼              ▼
 Flat Labels      Hierarchical
                  Label Smoothing
      │              │
      └──────┬───────┘
             ▼
     Performance Evaluation
```

---

## Results

<!-- TODO:
Replace with confusion matrices, qualitative examples, WandB plots,
or AP comparison figures.
-->

Example outputs may include:

- Detection examples
- Precision–Recall curves
- Confusion matrices
- Ablation plots
- Per-class AP comparison
- Training curves

---

## Limitations

The project intentionally reports both the strengths and limitations of Hierarchical Label Smoothing.

Current limitations include:

- Performance is highly sensitive to HLS hyperparameters.
- Large semantic categories receive weaker hierarchical supervision.
- Rare classes remain statistically underpowered due to limited validation samples.
- The QCVN 41 taxonomy reflects regulatory semantics rather than visual similarity, which may limit the effectiveness of hierarchy-based supervision.

---

## Future Work

Potential directions include:

- Adaptive category-size-aware label smoothing.
- Hierarchies based on visual similarity instead of regulatory categories.
- Hierarchical cosine loss.
- Evaluation on larger Vietnamese traffic sign datasets.
- Extension to newer YOLO architectures.

---

## Citation

If you find this repository useful, please consider citing:

```bibtex
@misc{phan2026hierarchical,
  title={Hierarchical Vietnamese Traffic Sign Detection using YOLO26n and Hierarchical Label Smoothing},
  author={Phan Van Tu},
  year={2026}
}
```

<!-- TODO:
Replace with your final paper citation after publication.
-->

---

## Acknowledgements

This project was developed as part of the Project-Based Learning (PBL) course at **FPT University**.

Built upon:

- Ultralytics YOLO26
- VNTS Dataset
- Weights & Biases
- QCVN 41 Vietnamese Road Traffic Sign Regulation
