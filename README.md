# Hierarchical Vietnamese Traffic Sign Detection with YOLO26n
### Incorporating QCVN 41 Semantic Hierarchies through Hierarchical Label Smoothing

> A research project investigating whether **Hierarchical Label Smoothing (HLS)** can improve Vietnamese traffic sign detection by leveraging the semantic hierarchy defined in the QCVN 41 road traffic sign regulation.

---

<!-- ===================================================================================== -->
<!-- TODO: Replace with a project banner (recommended size: ~1200 × 500)                   -->
<!-- Suggested content: Pipeline overview or representative detection result               -->
<!-- ===================================================================================== -->

<p align="center">
    <img src="assets/banner.png" width="90%">
</p>

<!-- ===================================================================================== -->
<!-- TODO: (Optional) Add a second figure showing qualitative detections or final results  -->
<!-- ===================================================================================== -->

<p align="center">
    <img src="assets/results_overview.png" width="90%">
</p>

---

# Project Description

Traffic sign detection is a fundamental component of Advanced Driver Assistance Systems (ADAS) and autonomous driving. Most object detectors formulate traffic sign recognition as a flat classification problem, treating every sign as an independent class despite many signs sharing similar semantic meanings. In Vietnam, however, traffic signs are formally organized under the **QCVN 41** regulation into five high-level categories: **Prohibitory, Warning, Mandatory, Information,** and **Supplementary**.

This project investigates whether incorporating this semantic hierarchy into the classification objective can improve object detection performance without modifying the detector architecture. We propose **Hierarchical Label Smoothing (HLS)**, a loss-level method that redistributes target probabilities among semantically related traffic signs while preserving the original YOLO26n architecture and inference pipeline. Extensive experiments on the **Vietnamese Traffic Signs (VNTS)** dataset compare HLS against a standard flat-label baseline through multiple hyperparameter ablations and statistical significance tests, revealing not only when hierarchical supervision improves performance, but also the conditions under which it becomes ineffective.

---

# Contributions

This project makes the following contributions:

- Introduces the first **QCVN 41 hierarchical annotation** for the VNTS dataset.
- Implements **Hierarchical Label Smoothing (HLS)** for YOLO26n without modifying the network architecture.
- Preserves identical inference speed by applying hierarchy only during training.
- Performs controlled comparisons against a flat-label baseline under identical training conditions.
- Conducts hyperparameter ablation on smoothing strength and classification loss weight.
- Identifies **semantic category size** as a major factor influencing the effectiveness of hierarchical supervision.
- Reports both the strengths and practical limitations of HLS through statistical significance analysis.

---

# Research Questions

This project aims to answer the following research questions:

1. Can Hierarchical Label Smoothing improve Vietnamese traffic sign detection compared to conventional flat-label training?

2. How do smoothing parameters and classification loss weighting influence HLS performance?

3. Which traffic sign categories benefit most from hierarchical supervision?

4. What limitations arise when using regulatory semantic hierarchies for object detection?

---

# Table of Contents

1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Method](#method)
4. [Experimental Results](#experimental-results)
5. [Usage](#usage)
6. [Repository Structure](#repository-structure)
7. [Limitations](#limitations)
8. [Future Work](#future-work)
9. [Citation](#citation)
10. [Acknowledgements](#acknowledgements)

---

# Installation

Clone the repository.

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```

Create a virtual environment (recommended).

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

Install the required dependencies.

```bash
pip install -r requirements.txt
```

<!-- ===================================================================================== -->
<!-- TODO: If your project requires a modified Ultralytics package, explain it here.       -->
<!-- Example: pip install -e .                                                              -->
<!-- ===================================================================================== -->

---

# Dataset

Experiments were conducted on the **Vietnamese Traffic Signs (VNTS)** dataset, publicly available on Kaggle.

| Property | Value |
|----------|-------:|
| Images | 3,216 |
| Annotated Instances | 8,334 |
| Traffic Sign Classes | 52 |
| Hierarchy Levels | 2 |
| Semantic Categories | 5 |
| Train / Validation Split | 2,552 / 639 |

For this project, every traffic sign class was mapped to the **QCVN 41** taxonomy, resulting in five semantic categories:

- Prohibitory
- Warning
- Mandatory
- Information
- Supplementary

Unlike the original flat-label dataset, this hierarchy enables semantically related traffic signs to share supervision during training through Hierarchical Label Smoothing.

> **Dataset:**  
> <!-- TODO: Insert Kaggle dataset link -->

> **Hierarchy Mapping:**  
> <!-- TODO: Link hierarchy mapping (.csv/.json/.yaml) if available -->

<!-- ===================================================================================== -->
<!-- TODO: Insert a sample annotated VNTS image here.                                      -->
<!-- ===================================================================================== -->

<p align="center">
    <img src="assets/sample_dataset.png" width="80%">
</p>

---

# Method

Rather than modifying the YOLO26n detector itself, this project introduces hierarchy exclusively within the classification loss.

The overall workflow consists of:

```
VNTS Dataset
      │
      ▼
Dataset Analysis
      │
      ▼
QCVN 41 Hierarchy Mapping
      │
      ▼
YOLO26n Training
      │
      ├───────────────┐
      ▼               ▼
 Flat Labels      Hierarchical
                  Label Smoothing
      │               │
      └──────┬────────┘
             ▼
      Model Evaluation
```

Hierarchical Label Smoothing redistributes the classification target probability from the ground-truth class toward semantically related sibling classes while leaving the detector architecture, bounding box regression, and inference procedure unchanged. Consequently, both the baseline model and the HLS model share identical computational complexity during deployment.

<!-- ===================================================================================== -->
<!-- TODO: Replace with your pipeline figure.                                              -->
<!-- ===================================================================================== -->

<p align="center">
    <img src="assets/pipeline.png" width="90%">
</p>

---

# Experimental Results

Five model configurations were evaluated.

| Model | α | β | λcls | mAP@0.5 |
|:------|--:|--:|------:|---------:|
| Baseline | — | — | 0.5 | **0.9487** |
| HLS | 0.05 | 0.10 | 0.5 | 0.8922 |
| HLS (Wide Gap) | 0.01 | 0.20 | 0.5 | 0.8735 |
| HLS (High Classification Weight) | 0.05 | 0.10 | 1.5 | 0.9219 |
| **HLS (Combined)** | **0.01** | **0.20** | **1.5** | **0.9583** |

### Main Findings

- The best-performing configuration achieved **95.83% mAP@0.5**, outperforming the flat-label baseline.
- Global improvement is statistically significant (*p* = 0.0265, Wilcoxon signed-rank test).
- Increasing the classification loss weight is essential for hierarchical supervision to become effective.
- Performance improvements correlate more strongly with **semantic category size** than with class frequency.
- Hierarchical supervision is highly sensitive to hyperparameter selection; poorly configured HLS may degrade detection performance.

<!-- ===================================================================================== -->
<!-- TODO: Insert result figures (recommended)                                             -->
<!-- 1. Baseline vs HLS comparison                                                         -->
<!-- 2. Hyperparameter ablation                                                            -->
<!-- 3. Per-category improvement                                                           -->
<!-- ===================================================================================== -->

<p align="center">
    <img src="assets/ablation.png" width="48%">
    <img src="assets/category_analysis.png" width="48%">
</p>

---

# Usage

### Train the Flat Baseline

```bash
# TODO: Replace with your actual training command
python train_baseline.py
```

### Train with Hierarchical Label Smoothing

```bash
# TODO: Replace with your actual training command
python train_hls.py
```

### Evaluate a Trained Model

```bash
# TODO: Replace with your evaluation command
python evaluate.py
```

### Reproduce the Experiments

<!-- ===================================================================================== -->
<!-- TODO: Explain how each experimental configuration (Baseline, A, A_v2, A_v3, A_v4)    -->
<!-- can be reproduced.                                                                    -->
<!-- ===================================================================================== -->

---

# Repository Structure

```text
project/
│
├── assets/                # Figures used in README
├── configs/               # Dataset and experiment configurations
├── datasets/              # Dataset utilities
├── notebooks/             # EDA and visualization notebooks
├── scripts/               # Training and evaluation scripts
├── ultralytics/           # Modified YOLO26 implementation
├── results/               # Experimental outputs
├── paper/                 # Research paper
├── report/                # Project report
├── requirements.txt
└── README.md
```

---

# Limitations

This work intentionally reports both the strengths and limitations of Hierarchical Label Smoothing.

Current limitations include:

- Performance is highly sensitive to HLS hyperparameters.
- Large semantic categories receive weaker hierarchical supervision because probability mass is distributed among more sibling classes.
- The QCVN 41 hierarchy reflects regulatory semantics rather than visual similarity.
- Rare classes contain relatively few validation samples, limiting statistical power.
- The experiments were conducted on a single Vietnamese traffic sign dataset; generalization to larger datasets remains unverified.

---

# Future Work

Potential directions for future research include:

- Adaptive category-aware smoothing weights.
- Hierarchies based on visual similarity instead of regulatory semantics.
- Hierarchical cosine loss and alternative hierarchical objectives.
- Evaluation on larger and more diverse Vietnamese traffic sign datasets.
- Extension to future YOLO architectures.

---

# Citation

If you find this repository useful, please consider citing:

```bibtex
@misc{yourcitation,
    title={Hierarchical Vietnamese Traffic Sign Detection using YOLO26n and Hierarchical Label Smoothing},
    author={Your Name},
    year={2026}
}
```

<!-- ===================================================================================== -->
<!-- TODO: Replace with your final publication citation.                                   -->
<!-- ===================================================================================== -->

---

# Acknowledgements

This project was completed as part of the **Project-Based Learning (PBL)** curriculum at **FPT University**.

This work builds upon:

- Ultralytics YOLO26
- Vietnamese Traffic Signs (VNTS) Dataset
- Weights & Biases
- QCVN 41: National Technical Regulation on Road Traffic Signs
