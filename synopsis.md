# KOLHAPUR INSTITUTE OF TECHNOLOGY's
## College of Engineering Kolhapur
### (Empowered Autonomous)
**R. S. No. 199B/1-3 Gokul Shirgaon, Kolhapur - 416 234**

---

# Project Synopsis

On

**NeuroStage**
**Lightweight CNN-Transformer Framework for Real-Time EEG-Based Sleep Stage Classification**

Submitted in partial fulfilment of the requirements for the
**Project Phase I**

---

**Submitted By**

| Sr. No. | Student Name | PRN |
| :--- | :--- | :--- |
| 1 | Pankaj Govekar | 2324000305 |
| 2 | Anurag Ilake | 2324000340 |
| 3 | Sandesh Salavi | 2324000347 |

**Under the Guidance of**

Mr. Rajesh Kumar Singh

**Department of Computer Science and Engineering**
**(Artificial Intelligence & Machine Learning) / (Data Science)**

**KOLHAPUR INSTITUTE OF TECHNOLOGY's**
**College of Engineering Kolhapur**

**2026–2027**

---

### Department of CSE (Artificial Intelligence and Machine Learning)
### Project Group Details

| Sr. No. | Student Name | Class | Roll No. | PRN |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Pankaj Govekar | B.Tech AIML | 29 | 2324000305 |
| 2 | Anurag Ilake | B.Tech AIML | 34 | 2324000340 |
| 3 | Sandesh Salavi | B.Tech AIML | 37 | 2324000347 |

**Group No.:** A14

**Project Title:** NeuroStage – Lightweight CNN-Transformer Framework for Real-Time EEG-Based Sleep Stage Classification

**Guide Name:** Mr. Rajesh Kumar Singh

**Sponsored By (if any):** ______________________________

**Co-Guide Name (if any) (Internal/External):** ______________________________

---

## 1. Project Title

NeuroStage – Lightweight CNN-Transformer Framework for Real-Time EEG-Based Sleep Stage Classification

## 2. Introduction

NeuroStage is a deep-learning-based sleep monitoring system designed to automatically classify the stages of sleep from EEG (Electroencephalography) signals.

The system focuses on five sleep stages:
* Wake
* N1
* N2
* N3
* REM

The main idea is to take a raw EEG signal, divide it into standard 30-second sleep epochs, and automatically predict the sleep stage for each epoch.

The project combines two important deep-learning ideas:
1. A lightweight 1D Convolutional Neural Network (1D-CNN) for learning useful waveform patterns from the EEG signal.
2. A compact multi-head self-attention/Transformer encoder for understanding the relationship between consecutive 30-second epochs.

The CNN is intended to learn time-frequency waveform patterns such as sleep spindles, K-complexes and delta-wave-related patterns. The Transformer then looks at the sequence of extracted epoch representations and learns temporal relationships between successive sleep epochs.

A major focus of NeuroStage is efficiency. Instead of building only a highly accurate but computationally heavy model, the project aims to build a model that is small and fast enough for real-time or near-real-time use on constrained hardware, wearables and edge devices.

The system also includes:
* Confidence-based hypnogram generation.
* Attention-weight visualization for interpretability.
* Accuracy-versus-efficiency benchmarking.
* Model compression experiments using pruning, INT8 quantization and knowledge distillation.
* A real-time inference/demo pipeline.

The project is based on public PSG datasets, particularly Sleep-EDF / Sleep-EDF Expanded and ISRUC-Sleep, and compares the proposed approach with established models such as DeepSleepNet, TinySleepNet and AttnSleep.

## 3. Problem Statement

Traditional sleep staging is commonly performed using Polysomnography (PSG). It requires overnight monitoring, multiple physiological channels such as EEG, EOG and EMG, and trained technicians who manually score sleep in 30-second epochs.

This creates several problems:
* Overnight laboratory monitoring can be costly.
* It is not convenient for continuous or home-based monitoring.
* Multi-channel setups can be difficult for wearable or low-power devices.
* Manual scoring requires trained experts.
* Different scorers can disagree, creating inter-scorer variability.

Existing deep-learning approaches may require large models, multiple channels and considerable computational resources. Such heavy models are difficult to deploy on low-power edge or wearable hardware.

Therefore, there is a need for a lightweight deep-learning system that can classify sleep stages from single-channel or minimal-channel EEG while maintaining competitive accuracy and reducing computational requirements.

NeuroStage addresses this problem by combining a lightweight depthwise-separable 1D-CNN with a compact attention/Transformer temporal encoder and by explicitly evaluating the trade-off between accuracy, parameter count and inference latency.

## 4. Why This Project Is Necessary

The project is necessary because there is a gap between accuracy and practical deployment.

A very large model may provide good classification performance but may not be suitable for a wearable or low-power device. On the other hand, a very small model may be efficient but can lose accuracy, particularly for difficult sleep stages such as N1.

NeuroStage attempts to find a practical middle ground:
**Good classification performance + small model + low latency + interpretability + deployment potential**

The project is therefore not simply another sleep-stage classifier. Its main research and engineering focus is the accuracy-versus-efficiency trade-off for real-time EEG sleep staging.

## 5. Objectives

The main objectives are:

**Objective 1 – Lightweight EEG Feature Extraction**
Design a lightweight 1D-CNN feature extractor using depthwise-separable convolutions to learn time-frequency patterns directly from raw single-channel EEG epochs.

**Objective 2 – Temporal Understanding**
Integrate a compact multi-head self-attention/Transformer encoder to learn dependencies between successive sleep epochs and sleep-stage transition patterns.

**Objective 3 – Model Compression**
Explore pruning, INT8 quantization and knowledge distillation to reduce model size and computational cost.

**Objective 4 – Dataset-Based Evaluation**
Evaluate the model using public PSG datasets such as Sleep-EDF / Sleep-EDF Expanded and ISRUC-Sleep.

**Objective 5 – Performance Measurement**
Measure Accuracy, Macro-F1, Cohen's Kappa, and Per-class F1.

**Objective 6 – Baseline Comparison**
Compare NeuroStage with established models such as DeepSleepNet, TinySleepNet, and AttnSleep. The comparison should focus not only on classification performance but also on parameter count, efficiency and latency.

**Objective 7 – Real-Time Demonstration**
Create a real-time or near-real-time inference pipeline that can process streamed EEG epochs and display the predicted sleep stages.

## 6. Literature Survey

Several existing studies have addressed automatic sleep-stage classification:

* **DeepSleepNet:** Supratak et al. proposed DeepSleepNet, which uses parallel CNN branches together with a bidirectional LSTM for automatic sleep staging from raw single-channel EEG.
* **TinySleepNet:** TinySleepNet simplified the approach and focused on reducing the number of parameters, making it more suitable for lightweight sleep staging.
* **AttnSleep:** AttnSleep combines multi-resolution CNN processing with multi-head attention to model temporal context in sleep-stage classification.
* **LWSleepNet:** LWSleepNet uses depthwise-separable multi-resolution convolutions and patch-based attention to reduce parameters and computation.
* **SleepTransformer:** SleepTransformer applies self-attention to spectrogram representations and also focuses on interpretability and uncertainty.

**Research Gap:**
The literature indicates that CNN and attention-based models can automate sleep staging directly from EEG. However, the project identifies an important gap: High-accuracy models can be computationally heavy, while lightweight models can lose accuracy (especially for N1). Few approaches explicitly report parameter count and latency for embedded/edge deployment. NeuroStage is designed to address this gap.

## 7. Proposed Methodology

**Step 1 – EEG Data Collection:** Public PSG datasets (Sleep-EDF, ISRUC-Sleep).
**Step 2 – Signal Preprocessing:** Band-pass filtering, z-score normalization, 30-second epoch segmentation, and labeling.
**Step 3 – Handling Class Imbalance:** Class-weighted loss for minority sleep stages (e.g., N1).
**Step 4 – CNN Feature Extraction:** Multi-resolution 1D-CNN with depthwise-separable convolutions.
**Step 5 – Temporal Transformer:** Compact multi-head self-attention/Transformer encoder (2–4 layers) for inter-epoch relationships.
**Step 6 – Classification:** Dense classification layer followed by Softmax.
**Step 7 – Lightweighting:** Pruning, INT8 Quantization, and Knowledge Distillation.
**Step 8 – Evaluation:** Subject-independent k-fold cross-validation.
**Step 9 – Hypnogram and Visualization:** Visualization of predicted sleep stages, confidence, and attention weights.
**Step 10 – Real-Time Demonstration:** Pipeline to process streamed EEG epochs.

## 8. System Architecture / Modules

```
EEG / PSG Dataset
↓
Data Loading & Band-Pass Filtering
↓
Z-Score Normalization
↓
30-Second Epoch Segmentation & Labelling
↓
Lightweight 1D-CNN
↓
Epoch Embeddings
↓
Multi-Head Self-Attention / Transformer Encoder
↓
Dense + Softmax Classification Head
↓
Wake / N1 / N2 / N3 / REM
↓
Confidence + Hypnogram + Attention Visualization
↓
Real-Time/Demo Dashboard
```

**Parallel model-optimization path:**
Trained Model → Pruning / INT8 Quantization / Knowledge Distillation → Smaller & Faster Model → Latency + Parameter Benchmarking → Edge/Wearable Deployment Potential

## 9. Major Modules

1. **Data Pipeline:** Load EEG, filter, normalize, segment into 30-sec epochs, assign labels, handle class imbalance.
2. **1D-CNN Feature Extraction:** Process raw epochs using multi-resolution depthwise-separable convolutions to produce compact embeddings.
3. **Attention/Transformer Temporal Encoder:** Apply multi-head self-attention to learn relationships between consecutive epochs.
4. **Classification Head:** Dense classification with Softmax to produce 5-class probabilities.
5. **Compression Module:** Test pruning, INT8 quantization, and knowledge distillation.
6. **Deployment/Demo Module:** Real-time inference, hypnogram generation, attention visualization, and benchmarking dashboard.

## 10. Expected Outcomes

* Develop a lightweight CNN-Transformer model for automatic sleep staging.
* Achieve competitive accuracy (~80–85%+ target on Sleep-EDF).
* Obtain Macro-F1 performance comparable to AttnSleep/TinySleepNet.
* Reduce parameter count and inference latency compared to heavier baselines.
* Provide an accuracy-versus-efficiency benchmark.
* Deliver a real-time or near-real-time inference pipeline for streamed EEG.
* Generate attention-weight visualizations and hypnograms.

## 11. Future Scope

1. **Full Edge Deployment:** Deploy on ARM Cortex-M, Raspberry Pi, or consumer EEG headbands and measure power consumption.
2. **Multi-Modal Input:** Combine EEG with EOG, EMG, or PPG/heart rate.
3. **Personalization:** Transfer learning to adapt the model to an individual's EEG patterns.
4. **Federated Learning:** Improve models across home devices without centralizing raw EEG data.
5. **Sleep Disorder Detection:** Extend research to sleep apnea, insomnia, or narcolepsy.
6. **Full Sleep-Health Application:** Build a mobile app for long-term sleep trend tracking.

## 12. References

1. Supratak, A., Dong, H., Wu, C., & Guo, Y. "DeepSleepNet: A Model for Automatic Sleep Stage Scoring based on Raw Single-Channel EEG." IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2017.
2. Supratak, A., & Guo, Y. "TinySleepNet: An Efficient Deep Learning Model for Sleep Stage Scoring based on Raw Single-Channel EEG." IEEE EMBC, 2020.
3. Eldele, E., et al. "An Attention-Based Deep Learning Approach for Sleep Stage Classification with Single-Channel EEG" (AttnSleep). IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2021.
4. Yang, C., Li, B., Li, Y., He, Y., & Zhang, Y. "LWSleepNet: A lightweight attention-based deep learning model for sleep staging with single-channel EEG." Digital Health / SAGE Journals, 2023.
5. Kemp, B., et al. "Sleep-EDF Database" – PhysioNet.
6. Khalighi, S., et al. "ISRUC-Sleep: A comprehensive public dataset for sleep researchers." Computer Methods and Programs in Biomedicine, 2016.
7. Iber, C., et al. "The AASM Manual for the Scoring of Sleep and Associated Events." American Academy of Sleep Medicine, 2007.
8. Vaswani, A., et al. "Attention Is All You Need." NeurIPS, 2017.

---

## Panel Remarks / Decision:

☐ Accepted
☐ Rejected
☐ Suggested Corrections

**Corrections / Suggestions:**
__________________________________________________________________________________
__________________________________________________________________________________

**Signatures:**

| Project Guide | Project Coordinator | HoD CSE(AIML) |
| :--- | :--- | :--- |
| Mr. Rajesh Kumar Singh | Mr. Avinash P. Budaragade | Dr. Uma P. Gurav |
