# End-to-End Medical AI System for Pneumonia Detection
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Motea99/Pneumonia-Diagnosis-System-Challenge/blob/main/MAI_End_To_End_Demo.ipynb)
Medical AI Pneumonia Diagnosis System

End-to-End Deep Learning, Visual Language Model, and Retrieval Prototype
AlfaisalX: Cognitive Robotics & Autonomous Agents – Technical Challenge

---

1. Project Overview

This project presents a complete end-to-end Artificial Intelligence system for automated analysis of chest X-ray images.
The system integrates three modern AI components:

1. Image classification (diagnosis)
2. Medical report generation (radiology-style text)
3. Content-based image retrieval (similar case search)

The objective is not only to train a neural network, but to demonstrate how multiple AI technologies can be combined into a functional medical decision-support prototype.

The system was developed using the PneumoniaMNIST dataset (MedMNIST v2) and implemented entirely in Python using PyTorch and HuggingFace models.

---

2. System Capabilities

Given a chest X-ray image, the system can:

• Detect whether pneumonia is suspected
• Visualize important regions using Explainable AI (Grad-CAM)
• Generate an automatic medical report using a visual-language model
• Retrieve visually similar X-ray cases from the dataset

This mimics a simplified real-world clinical workflow.

---

3. Dataset

Dataset: PneumoniaMNIST (MedMNIST v2)

- Binary classification: Normal vs Pneumonia
- Training images ˜ 4,700
- Validation images ˜ 500
- Test images ˜ 600
- Image size: 28×28 grayscale

Install automatically:

pip install medmnist

Official website: https://medmnist.com

---

4. Model Architecture (Task 1 – Classification)

We used a Vision Transformer (ViT) as the primary classifier.

Preprocessing

- CLAHE contrast enhancement
- Normalization
- Data augmentation (rotation, flip)

Training Configuration

- Epochs: 5
- Optimizer: Adam
- Loss Function: Cross-Entropy Loss
- Framework: PyTorch

Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve

Performance

Metric| Value
Accuracy| 86.38%
Precision| 82.24%
Recall| 99.74%
F1-Score| 90.15%

---

5. Explainable AI (Grad-CAM)

To improve interpretability, Grad-CAM heatmaps were generated to highlight regions influencing model predictions.
The model mainly focused on lung opacity regions in pneumonia cases, indicating clinically meaningful behavior.

---

6. Medical Report Generation (Task 2 – VLM)

A visual-language model (LLaVA / Med-style prompting) was used to automatically produce a radiology-style report.

The model receives:

- X-ray image
- Clinical prompt

and outputs a natural language medical description including suspicion of pneumonia.

Sample prompt:

«"You are a radiologist. Analyze this chest X-ray and write a concise medical report. State if pneumonia is suspected."»

---

7. Semantic Image Retrieval (Task 3 – CBIR)

A content-based image retrieval system is implemented using image embeddings and vector similarity search (FAISS).

Capabilities:

- Image-to-Image search
- Retrieval of most similar cases
- Visualization of top-k matches

This simulates a clinical “similar cases” reference system.

---

8. Repository Structure

medical-ai-pneumonia-diagnosis-system/
¦
+-- models/                     # Saved trained model
+-- notebooks/                  # Colab demo notebook
+-- task1_classification/       # Training and evaluation
+-- task2_report_generation/    # Medical report generation
+-- task3_retrieval/            # Similar image retrieval
+-- reports/                    # Written reports
+-- images/                     # Visual results
+-- requirements.txt

---

9. How to Run the Project

1. Clone repository

git clone https://github.com/YOUR_USERNAME/medical-ai-pneumonia-diagnosis-system.git
cd medical-ai-pneumonia-diagnosis-system

2. Install dependencies

pip install -r requirements.txt

3. Run demo notebook

Open:

notebooks/End_to_End_Demo.ipynb

or run directly in Google Colab.

---

10. Research Objectives

This project demonstrates:

- Integration of Deep Learning + Multimodal AI
- Explainable AI in medical imaging
- Automated medical report generation
- Retrieval-based clinical decision support

---

11. Author

Dr. Motea Alsamawi
Researcher in Medical AI & Biomedical Engineering

---

12. Notes

This repository was developed as part of a Postdoctoral Technical Challenge at Alfaisal University (MedX Research Unit).


The code is intended for research and educational purposes only and not for clinical diagnosis.









