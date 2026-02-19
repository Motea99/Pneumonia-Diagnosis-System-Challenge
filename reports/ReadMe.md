
End-to-End Medical AI System for Pneumonia Analysis Using Deep Learning, Visual Language Models, and Semantic Retrieval

Researcher’s Name:  Motea Abdulraqeb Abduljalil Alsamawi

Research Fellow / Postdoctoral Candidate

Alfaisal: Cognitive Robotics & Autonomous Agents – MedX Unit

18-2-2026

7-Day Technical Challenge Submission

1. Executive Summary

This project presents the development of an end-to-end artificial intelligence system for automated analysis of chest X-ray images with a focus on pneumonia detection. The proposed system integrates three complementary components: image-based diagnosis using deep learning, decision interpretability using explainable AI techniques, and automated clinical report generation using a visual-language model. In addition, the system is designed with the potential to support retrieval of visually similar cases, reflecting real-world clinical workflows.

A convolutional and transformer-based classification model was trained on the PneumoniaMNIST dataset to distinguish normal chest radiographs from pneumonia cases. Beyond classification accuracy, the system emphasizes reliability and transparency. Grad-CAM explainability was employed to visualize the regions of the lungs that influenced the model’s predictions, allowing verification that the model focuses on clinically relevant anatomical structures rather than spurious patterns.

To further approximate clinical practice, a medical visual language model was integrated to automatically generate short radiology-style reports from chest X-ray images. This component demonstrates how modern multimodal AI systems can bridge the gap between numerical predictions and human-interpretable medical communication.

Overall, the project demonstrates a prototype clinical decision-support pipeline in which a single input image can be analyzed, interpreted, and described in natural language. The objective is not to replace physicians but to assist radiologists by improving screening efficiency, highlighting suspicious regions, and providing preliminary documentation. The system therefore represents a step toward practical and explainable medical AI systems suitable for healthcare environments.

2. Problem Motivation & Clinical Context

Pneumonia remains one of the most common and clinically significant respiratory diseases worldwide and is a major cause of morbidity and hospital admissions. Early diagnosis is critical because delayed detection can lead to severe complications, particularly among elderly and immunocompromised patients. Chest X-ray imaging is the primary screening tool used in routine clinical practice due to its availability, speed, and relatively low cost.

However, interpreting chest radiographs is a complex task. Radiographic signs of pneumonia can be subtle and may overlap with other pulmonary conditions such as atelectasis, pulmonary edema, or imaging artifacts. Accurate interpretation requires significant expertise, and even experienced radiologists may occasionally disagree, especially under high workload conditions. In many healthcare systems, the increasing number of imaging studies combined with a limited number of radiology specialists leads to reporting delays and diagnostic variability.

Artificial intelligence has emerged as a promising solution for supporting radiologists in image interpretation. Deep learning models can learn discriminative visual patterns from large collections of medical images and assist in detecting abnormalities. Nevertheless, clinical adoption of AI systems requires more than high classification accuracy. Physicians must understand why a system makes a decision, and outputs must be communicated in a clinically meaningful manner.

For this reason, a practical medical AI system should not only provide a prediction but also an explanation and an interpretable report. Integrating image classification, explainable AI visualization, and natural-language report generation can transform AI from a standalone algorithm into a clinical decision-support tool. The present work addresses this need by proposing an integrated pipeline that analyzes chest X-ray images, highlights relevant pathological regions, and produces a preliminary radiology-style description that aligns with clinical workflow.

3. System Overview (Architecture)

The proposed system is an end-to-end medical AI pipeline designed to simulate a real clinical decision-support workflow. The architecture integrates three main components: an image classification module, a visual-language report generation module, and an explainability module.

First, a chest X-ray image is provided as input to the system. The image undergoes preprocessing, including resizing, normalization, and contrast enhancement using CLAHE to improve the visibility of lung structures. The processed image is then passed to the Vision Transformer (ViT) classifier, which predicts whether the image represents a normal case or pneumonia.

After classification, an explainability stage is applied using Grad-CAM. This module highlights the image regions that contributed most to the model’s decision, producing a heatmap overlay on the original X-ray. This step is critical for clinical trust, as it allows verification that the model focuses on anatomically meaningful lung regions rather than irrelevant image artifacts.

Next, the same image is forwarded to a visual language model (VLM), specifically a medical-adapted multimodal model. The VLM analyzes the radiographic patterns and generates a short textual medical report describing the findings and indicating whether pneumonia is suspected.

Finally, the outputs are presented together: the predicted diagnosis, confidence behavior, explainability heatmap, and generated radiology-style report. By combining automated diagnosis, visual explanation, and natural language reporting, the system emulates how an AI assistant could support radiologists in real-world clinical environments. Figure 1shows the Proposed System Architecture

 
Figure 1: Proposed End-to-End Medical AI Pipeline


Task 1 — Pneumonia Classification

4. Dataset Description

All experiments in this project were conducted using the PneumoniaMNIST dataset from the MedMNIST v2 benchmark. The dataset is a curated subset of pediatric chest X-ray images designed for binary classification between normal lungs and pneumonia cases. Each image is provided as a 28×28 grayscale radiograph, which makes the dataset computationally lightweight and suitable for rapid prototyping of medical AI systems.

The dataset is divided into three subsets: a training set of approximately 4,700 images, a validation set of about 500 images, and a test set of roughly 600 images. The labels are binary, where class 0 corresponds to normal and class 1 corresponds to pneumonia. This standardized benchmark ensures fair evaluation and reproducibility of results across different research implementations.

Although the image resolution is relatively low compared to real clinical radiographs, the dataset still preserves essential lung patterns such as opacities and consolidation regions that are characteristic of pneumonia. Therefore, it provides an appropriate environment for evaluating the feasibility of automated diagnostic pipelines before scaling to higher-resolution clinical datasets.


5. Preprocessing and data preparation

Before training the model, several preprocessing steps were applied to prepare the images for deep learning analysis. Since the original images are grayscale and of low resolution, they were resized to match the input requirements of the selected architecture. The images were converted to three-channel format to ensure compatibility with pretrained computer vision models.

Contrast Limited Adaptive Histogram Equalization (CLAHE) was applied to enhance local contrast within the lung fields. Medical X-ray images often suffer from low contrast, making pathological regions difficult to distinguish. The CLAHE technique improves the visibility of subtle opacities by amplifying local intensity variations while avoiding over-amplification of noise. This step helps the model focus on clinically meaningful structures rather than background artifacts.

Normalization was also applied to scale pixel intensities into a stable range suitable for neural network training. This reduces gradient instability and improves convergence during optimization. The dataset was then organized into training and testing loaders with batch processing to enable efficient model training and evaluation.


6  Model Architecture

For the classification task, a Vision Transformer (ViT) architecture was selected. Unlike conventional convolutional neural networks that rely on local receptive fields, Vision Transformers process images as sequences of image patches and learn global contextual relationships using self-attention mechanisms. This property is particularly useful in medical imaging, where disease patterns may extend across multiple lung regions rather than appearing as isolated local features.

A pretrained Vision Transformer model was fine-tuned on the PneumoniaMNIST dataset. Transfer learning was employed to leverage prior knowledge learned from large-scale image datasets, enabling effective learning despite the relatively small size of the medical dataset. Only the final classification layers were adapted to the binary classification task.

The model was trained for 5 epochs using a supervised learning framework. A cross-entropy loss function was used for optimization, as it is appropriate for binary classification problems. During training, the model parameters were updated iteratively to minimize classification error on the training data while maintaining generalization to unseen test images.

The relatively small number of training epochs was intentionally selected to prevent overfitting due to the limited dataset size. The objective was to obtain a stable and generalizable model rather than memorizing the training images. The trained model weights were saved to allow reproducibility and reuse without retraining.

7  Evaluation & Results

The trained Vision Transformer model was evaluated on the independent test set to assess its generalization capability. Several standard classification metrics were calculated, including accuracy, precision, recall, and F1-score.

The model achieved an overall accuracy of 86.38%, indicating a reliable performance in distinguishing between normal and pneumonia chest X-ray images. The precision reached 82.24%, while the recall was significantly higher at 99.74%, resulting in an F1-score of 90.15%.

A detailed class-wise analysis provides deeper insights into the model behavior. For pneumonia detection, the model demonstrated excellent sensitivity, achieving a recall of 1.00 (389 out of 390 pneumonia cases correctly detected). This means that almost all infected cases were successfully identified. From a clinical perspective, this is highly desirable because missing pneumonia cases (false negatives) could lead to severe medical consequences.

However, the performance for normal cases was lower, with a recall of 0.64. The confusion matrix shows that 84 normal images were misclassified as pneumonia, while only a single pneumonia case was misclassified as normal. Therefore, the model exhibits a strong bias toward detecting pneumonia.

Confusion Matrix:
Normal predicted as Normal: 150
Normal predicted as Pneumonia: 84
Pneumonia predicted as Normal: 1
Pneumonia predicted as Pneumonia: 389

This behavior suggests that the classifier is operating as a highly sensitive screening tool. In medical diagnostics, such behavior is often acceptable and even desirable, as false positives can be resolved by a radiologist, while false negatives (missed disease) are much more dangerous. The model therefore prioritizes patient safety by minimizing missed pneumonia cases.

The weighted average F1-score of 0.86 confirms that the model maintains a good balance between precision and recall across the dataset. The high recall for pneumonia indicates that the Vision Transformer successfully learned relevant pathological patterns in lung regions despite the low image resolution of the dataset.

Overall, the results demonstrate that the proposed pipeline can serve as a preliminary decision-support system, assisting clinicians in identifying suspected pneumonia cases while reducing the likelihood of overlooked infections.

8 Evaluation metrics:

Accuracy	Precision	Recall	F1-Score
86.38%	82.24%	99.74%	90.15%

(Class-wise Performance):

Class	Precision	Recall	F1-Score	Support
Normal	0.99	0.64	0.78	234
Pneumonia	0.82	1.00	0.90	390

 9 Error Analysis 

To better understand the behavior of the proposed model beyond aggregate metrics, a detailed error analysis was conducted using the confusion matrix and misclassified samples. The results reveal a clear asymmetry in the model’s errors. While the model demonstrates extremely high sensitivity for pneumonia detection (recall ≈ 1.00), most classification mistakes occur in the normal class. Specifically, a considerable number of normal chest X-ray images were incorrectly predicted as pneumonia (false positives), whereas only a single pneumonia case was classified as normal (false negative).

This behavior indicates that the model learned a conservative diagnostic strategy, prioritizing detection of pathological patterns over preserving specificity. From a clinical perspective, this tendency is understandable: missing a pneumonia case may lead to serious medical consequences, whereas a false alarm typically results only in additional examination. The Grad-CAM visualizations support this observation, as the model frequently attends to high-contrast lung regions and subtle texture irregularities that may resemble infiltrates, even in otherwise normal images.

Several factors may explain these errors. First, the PneumoniaMNIST images are low-resolution (28×28), which removes fine anatomical details and forces the model to rely on coarse texture patterns rather than clear radiological structures. Second, variations in illumination and rib-shadow intensity can mimic opacities associated with pneumonia. Third, the class distribution and the use of contrast enhancement (CLAHE) amplify subtle density differences, which improves sensitivity but may also increase false positives.

Overall, the model behaves as a high-recall screening system rather than a strict diagnostic classifier. It is therefore more suitable as an early detection assistant that flags suspicious cases for further review by a clinician. Future improvements could focus on increasing specificity by incorporating higher-resolution images, lung region segmentation, or hybrid clinical features to reduce over-sensitivity to benign structural variations.

XAI Interpretation – Grad-CAM Analysis

The Grad-CAM visualizations demonstrate that the Vision Transformer model focuses predominantly on anatomically relevant pulmonary regions when predicting pneumonia.

Across the presented samples:

The attention maps highlight the lower lung zones and perihilar regions, which are clinically common locations for pneumonic infiltrates.

In samples with suspected pneumonia, the model concentrates on areas showing increased opacity and patchy consolidations.

The activation heatmaps align with radiologically meaningful structures rather than irrelevant regions (e.g., borders, background, labels), suggesting that the model learned clinically valid spatial patterns.

The attention is bilaterally distributed in some cases, consistent with diffuse or multifocal infection patterns.

These findings indicate that the model's decision-making process is not random but anatomically grounded, enhancing the interpretability and clinical reliability of the system.

Overall, the Grad-CAM results support that the ViT-based classifier captures pathologically significant radiographic features associated with pneumonia.
