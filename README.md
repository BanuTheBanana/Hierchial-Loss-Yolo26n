<img width="976" height="161" alt="image" src="https://github.com/user-attachments/assets/e90aea77-b7ee-4fb2-a38e-23077eea5e8a" /># Hierchial-Loss-Yolo26n


<img width="1920" height="1100" alt="val_batch2_labels" src="https://github.com/user-attachments/assets/d768eb27-515f-496f-8c70-9671617ce575" />


# Hierarchical Vietnamese Traffic Sign Detection using YOLO26n

> Improving Vietnamese traffic sign detection by incorporating the QCVN 41 traffic sign hierarchy into YOLO26n through Hierarchical Label Smoothing.

<p align="center">
  <!-- TODO: Pipeline Diagram -->
  <img width="976" height="161" alt="Screenshot 2026-07-11 124015" src="https://github.com/user-attachments/assets/fca7fae5-8457-430a-8567-279c99b27b40" /> 
</p>

<p align="center">
  <!-- TODO: Detection Result -->
  <img src="images/demo.png" width="85%">
</p>

## Overview

Traffic sign detection is a fundamental component of Advanced Driver Assistance Systems (ADAS) and autonomous driving. While most object detectors treat every traffic sign as an independent class, Vietnamese traffic signs are organized under the QCVN 41 regulation into five semantic categories: Prohibitory, Warning, Mandatory, Information, and Supplementary. This project investigates whether incorporating this hierarchical structure into the training objective can improve detection performance. Rather than modifying the detector architecture, we introduce **Hierarchical Label Smoothing (HLS)**, which redistributes the classification target probability toward semantically related sibling classes while preserving the original YOLO26n inference pipeline. Through experiments on the VNTS dataset and multiple ablation studies, we evaluate when hierarchical supervision is beneficial and analyze the factors that influence its effectiveness.

## Key Features

This project provides the first hierarchical annotation framework for the Vietnamese Traffic Signs (VNTS) dataset based on the QCVN 41 taxonomy, implements Hierarchical Label Smoothing for YOLO26n without introducing additional inference cost or architectural modifications, performs comprehensive benchmarking against a flat-label baseline, includes hyperparameter ablation on smoothing and classification loss weighting, and presents statistical analyses that explain how category size influences the effectiveness of hierarchical supervision.
