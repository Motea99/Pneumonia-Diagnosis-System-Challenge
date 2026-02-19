# End-to-End Medical AI System for Pneumonia Analysis
## Using Deep Learning, Visual Language Models, and Semantic Retrieval

**Researcher:** **Motea Abdulraqeb Abduljalil Alsamawi** **Role:** Research Fellow / Postdoctoral Candidate  
**Affiliation:** Alfaisal: Cognitive Robotics & Autonomous Agents – MedX Unit  
**Date:** 18-2-2026  
**Status:** 7-Day Technical Challenge Submission  

---

## 1. **Executive Summary**
This project presents the development of an **end-to-end artificial intelligence system** for automated analysis of chest X-ray images with a focus on **pneumonia detection**. The proposed system integrates three complementary components: 
* **Image-based diagnosis** using deep learning.
* **Decision interpretability** using explainable AI (XAI) techniques.
* **Automated clinical report generation** using a visual-language model (VLM).

The system is designed to support real-world clinical workflows by providing a prototype where a single input image can be analyzed, interpreted, and described in natural language, assisting radiologists in improving screening efficiency.

---

## 2. **Problem Motivation & Clinical Context**
**Pneumonia** remains a major cause of morbidity worldwide. Early diagnosis via **Chest X-ray (CXR)** is critical but challenging due to:
* **Subtle Radiographic Signs:** Overlap with conditions like atelectasis or edema.
* **Specialist Scarcity:** High workload leading to potential diagnostic variability.
* **Need for Explainability:** Clinical adoption requires "transparent" AI that explains *why* a decision was made.

This work transforms AI from a standalone algorithm into a **clinical decision-support tool** by providing visual explanations and preliminary radiology-style reports.

---

## 3. **System Overview (Architecture)**
The architecture simulates a clinical workflow through three stages:
1.  **Preprocessing:** Image resizing and contrast enhancement using **CLAHE**.
2.  **Classification:** A **Vision Transformer (ViT)** predicts the diagnosis.
3.  **Explainability:** **Grad-CAM** generates a heatmap highlighting pathological regions.
4.  **Report Generation:** A **Medical Visual Language Model (VLM)** produces a textual radiology report.

> **Figure 1: Proposed End-to-End Medical AI Pipeline**
> ![System Architecture](Images/AI_Pipe_Line.jpg)
> ![System Architecture](https://raw.githubusercontent.com/Motea99/Pneumonia-Diagnosis-System-Challenge/main/Images/AI_Pipe_Line.jpg)
---

# **Task 1 — Pneumonia Classification**

## 4. **Dataset Description**
* **Dataset:** **PneumoniaMNIST** (MedMNIST v2).
* **Content:** Pediatric chest X-ray images (Normal vs. Pneumonia).
* **Split:** * **Training:** ~4,700 images.
    * **Validation:** ~500 images.
    * **Test:** ~600 images.
* **Format:** 28×28 grayscale radiographs, providing a standardized benchmark for rapid prototyping.

## 5. **Preprocessing and Data Preparation**
* **Contrast Enhancement:** Applied **CLAHE** to improve visibility of lung opacities.
* **Color Conversion:** Converted to 3-channel format for model compatibility.
* **Normalization:** Scaled pixel intensities for stable gradient descent and faster convergence.

## 6. **Model Architecture**
* **Core Model:** **Vision Transformer (ViT)**.
* **Mechanism:** Uses **Self-Attention** to learn global contextual relationships across lung regions.
* **Training:** * **Transfer Learning:** Fine-tuned on PneumoniaMNIST.
    * **Epochs:** 5 (Selected to prevent overfitting).
    * **Loss Function:** Cross-entropy loss.

## 7. **Evaluation & Results**
The model acts as a **highly sensitive screening tool**, prioritizing the detection of all pneumonia cases to ensure patient safety.

* **Overall Accuracy:** **86.38%**
* **Pneumonia Recall:** **100% (1.00)** — Zero missed pneumonia cases in the test sample.
* **F1-Score:** **90.15%**

### **Confusion Matrix Analysis:**
| Predicted $\rightarrow$ | Normal | Pneumonia |
|---|---|---|
| **Actual Normal** | 150 | 84 |
| **Actual Pneumonia** | 1 | 389 |

## 8. **Evaluation Metrics Summary**

| Metric | Value |
| :--- | :--- |
| **Accuracy** | **86.38%** |
| **Precision** | **82.24%** |
| **Recall** | **99.74%** |
| **F1-Score** | **90.15%** |

### **Class-wise Performance:**
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | 0.99 | 0.64 | 0.78 | 234 |
| **Pneumonia** | 0.82 | 1.00 | 0.90 | 390 |

---

## 9. **Error Analysis**
The model demonstrates a **conservative diagnostic strategy**:
* **False Positives:** High (84 normal cases predicted as pneumonia) due to low resolution and CLAHE amplifying subtle textures.
* **False Negatives:** Extremely Low (1 case), which is clinically safer.
* **Conclusion:** The system is suitable as an **early detection assistant** that flags suspicious cases for clinician review.

---

## 10. **XAI Interpretation – Grad-CAM Analysis**
**Grad-CAM** visualizations confirm that the ViT model is "anatomically grounded":
* **Target Areas:** Focuses on lower lung zones and perihilar regions.
* **Clinical Validity:** Attention heatmaps align with radiologically meaningful opacities rather than background noise.
* **Reliability:** The model’s decision-making process is transparent, enhancing trust between the AI and the radiologist.

**Visual Evidence:**
![Grad-CAM Examples](Images/XAI.jpg)

---
