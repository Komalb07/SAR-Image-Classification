# SAR Image Classification

This repository contains the implementation of a **Synthetic Aperture Radar (SAR) Image Classification** pipeline. The project explores **image filtering techniques**, **feature enhancement**, and **machine learning-based classification** to segment SAR images into meaningful categories: **urban, vegetation, and water**.

## Project Overview

Synthetic Aperture Radar (SAR) images provide valuable data for **remote sensing applications** but suffer from noise, such as **speckle**, which can obscure meaningful information. This project addresses these challenges by:

- **Applying advanced filtering techniques** to enhance image quality.
- **Using machine learning models** for SAR image classification.
- **Evaluating model performance** using **Mean Squared Error (MSE) and classification accuracy**.

The goal is to **improve SAR image interpretability** and enable robust segmentation for **geospatial and environmental analysis**.

## Key Features

- **SAR Image Preprocessing**: Reads **HH and HV polarization bands**, enhances intensity, and performs multilooking.
- **Speckle Noise Reduction**: Implements **five filtering techniques** to improve image clarity.
- **Automated Image Segmentation**: Uses **Unsupervised (K-Means) and Supervised (SVM, Decision Tree, Random Forest) classifiers**.
- **Performance Evaluation**: Assesses **filter effectiveness (MSE)** and **classifier accuracy based on segmented outputs**.

## Methodology

### **1. Preprocessing & Feature Enhancement**
- Loaded SAR image bands (**HH and HV polarization**) using `GDAL`.
- Enhanced image intensity and performed **multilooking** for improved visualization.

### **2. Noise Reduction (Filtering)**
Applied the following **speckle noise filters**:
- **Boxcar Filter**: Averages pixel values within a moving window.
- **Refined Lee Filter**: Balances noise reduction with edge preservation.
- **Median Filter**: Removes noise while maintaining edges.
- **Mean Filter**: Applies a kernel-based averaging approach.
- **Gaussian Filter**: Smooths image using a Gaussian kernel.

### **3. Image Classification**
- **Unsupervised Learning**: Used **K-Means clustering** to segment SAR images.
- **Supervised Learning**: Implemented **Support Vector Machine (SVM), Decision Tree, and Random Forest Classifiers** to classify urban, vegetation, and water areas.

### **4. Model Evaluation**
- **Filtering Performance**: Measured using **Mean Squared Error (MSE)**.
- **Classification Accuracy**: Evaluated based on **segmentation quality** and classifier performance.

## Results

### **Filter Performance (MSE)**
| Filter            | Mean Squared Error (MSE) |
|------------------|-------------------------|
| Mean Filter      | *30.535*                 |
| Median Filter    | *36.116*                 |
| Gaussian Filter  | *35.177*                 |
| Refined Lee      | **14.0005** (Best)       |
| Boxcar Filter    | *29.559*                 |

**Key Insights:**
- **Refined Lee Filter** provided the **lowest MSE**, indicating **better noise suppression and feature preservation**.
- **Median and Gaussian Filters** retained more image details but resulted in **higher MSE** due to residual speckle noise.

### **Classification Results**
- **Unsupervised Learning** (K-Means) provided a **preliminary segmentation** but lacked class labels.
- **Supervised Learning**:
  - **SVM, Decision Tree, and Random Forest** successfully segmented images into **urban, vegetation, and water**.
  - **Random Forest performed best**, achieving **clearer segmentation and robustness to noise**.

## Requirements

- Python 3.8+
- Required Libraries:
  - `GDAL`
  - `Pillow`
  - `numpy`
  - `scipy`
  - `scikit-learn`
  - `matplotlib`
  - `astropy`

## Future Enhancements

- **Enhance Filtering Techniques**: Test adaptive and deep-learning-based noise reduction methods.
- **Improve Classification Accuracy**: Explore **CNN-based deep learning models** for SAR classification.
- **Real-Time Processing**: Optimize for **large-scale datasets** with **faster computation**.
