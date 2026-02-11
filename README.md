# Brain Tumor Classification using Deep Learning (CNN)

##  Project Overview
This project focuses on **brain tumor classification using MRI images** by applying **deep learning techniques**.  
The objective is to perform a ** advanced Convolutional Neural Network (CNN) architectures**, specifically **Xception** and **EfficientNet**, to evaluate their effectiveness in classifying different types of brain tumors.

This work was carried out as a **final semester project** and later resulted in a **published research paper**.


##  Problem Statement
Manual analysis of brain MRI scans is time-consuming and may vary based on human interpretation.  
This project aims to assist medical professionals by providing an **automated and consistent decision-support system** for brain tumor classification.


##  Objectives
- To classify brain MRI images into different tumor categories  
- To apply **deep learning–based CNN models** for medical image analysis  
- To analyze both model's performance with same training and testing dataset  

## 🧪 Dataset
- The dataset consists of **labeled brain MRI images**
- Each image belongs to one of the following classes:
  - Glioma
  - Meningioma
  - Pituitary Tumor
  - No Tumor
- The dataset is split into **training and testing sets**


##  Technologies & Tools Used
- **Python**
- **TensorFlow / Keras**
- **NumPy, Pandas**
- **Matplotlib**
- **Google Colab**

---

##  Deep Learning Models Used

### 1️. Xception
- An advanced CNN architecture based on **depthwise separable convolutions**
- Efficient in feature extraction with reduced computational complexity

### 2️. EfficientNet
- A CNN architecture that uses **compound scaling** of depth, width, and resolution
- Achieves high accuracy with fewer parameters


## 🔄 Methodology
1. Image preprocessing (resizing, normalization)
2. Use of **pre-trained CNN models** (transfer learning)
3. Addition of **custom classification layers**
4. Model training and fine-tuning
5. Performance evaluation and comparison


## 🔁 Transfer Learning Strategy
- Pre-trained weights were used as feature extractors
- Base layers were initially frozen
- Custom dense and dropout layers were added
- Fine-tuning was applied to adapt the models for brain tumor classification


##  Evaluation Metrics
- Accuracy
- Confusion Matrix
- Classification Report

Accuracy alone was not considered sufficient due to the **medical nature of the problem**.


##  Input &  Output
- **Input:** Brain MRI image  
- **Output:** Predicted tumor class (Glioma, Meningioma, Pituitary, or No Tumor)

---

##  Key Learnings
- Understanding of CNN-based deep learning models
- Practical experience with transfer learning and fine-tuning
- Comparative analysis of deep learning architectures
- Awareness of ethical considerations in medical AI

##  Disclaimer
This project is intended as a **decision-support tool** and should not be used as a replacement for professional medical diagnosis.


##  Publication
The results and analysis of this project were documented and published as a **research paper**.

##  Published Research Paper

This project is my **Final Semester Project** and is based on a **published research paper**.

**Title:** Brain Tumor Classification and Detection by Explainable AI, Xception and EfficientNet Models for Improving Performance Metrices 
**Journal / Conference:** 5th International Conference on Intelligent Technologies (CONIT 2025)  
**Year:** 2025  
**DOI / Paper Link:** [https://<official-paper-link>](https://ieeexplore.ieee.org/document/11167506)

