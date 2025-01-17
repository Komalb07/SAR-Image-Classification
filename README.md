# SAR-Image-Classification

This repository contains the implementation of a **Synthetic Aperture Radar (SAR) Image Classification** pipeline. The project explores image filtering techniques, feature enhancement, and classification methodologies to segment SAR images into meaningful categories such as **urban**, **vegetation**, and **water**.

## Project Overview

Synthetic Aperture Radar (SAR) images provide valuable data for remote sensing applications but often suffer from noise, such as speckle, that can obscure meaningful information. This project addresses these challenges by:

- Applying advanced filtering techniques to reduce noise.
- Implementing supervised and unsupervised machine learning models for classification.
- Evaluating the performance of each filter and classifier based on Mean Squared Error (MSE) and classification accuracy.

The techniques used aim to enhance SAR image interpretability and enable robust segmentation.

## Key Features

1. **Noise Reduction**:
   - Implemented filtering methods including:
     - Boxcar Filter
     - Refined Lee Filter
     - Median Filter
     - Mean Filter
     - Gaussian Filter
   - Calculated Mean Squared Error (MSE) to evaluate each filter’s effectiveness.

2. **Image Classification**:
   - **Unsupervised Learning**:
     - Used K-Means clustering to segment the image into predefined classes.
   - **Supervised Learning**:
     - Implemented classifiers like Support Vector Machines (SVM), Decision Trees, and Random Forests for accurate segmentation.

3. **Evaluation**:
   - Quantified filter performance using MSE values.
   - Assessed classifier performance based on visual results and segmented outputs.

## Methodology

### 1. Preprocessing
- Read SAR image bands (HH and HV polarization) using GDAL.
- Enhanced image intensity and performed multilooking for better visualization.

### 2. Filtering
- Applied the following filters to reduce speckle noise:
  - **Boxcar Filter**: Uses a simple moving average to smooth the image.
  - **Refined Lee Filter**: Balances noise reduction and feature preservation.
  - **Median Filter**: Retains edges while reducing noise.
  - **Mean Filter**: Uses a kernel-based averaging approach.
  - **Gaussian Filter**: Blurs the image using a Gaussian kernel.

### 3. Classification
- Performed **unsupervised classification** using K-Means clustering.
- Conducted **supervised classification** using:
  - Support Vector Machine (SVM)
  - Decision Tree Classifier
  - Random Forest Classifier
- Defined classes: Urban, Vegetation, and Water.

## Results

### Filter Performance
| Filter            | Mean Squared Error (MSE) |
|--------------------|---------------------------|
| Mean Filter        | *30.535*                  |
| Median Filter      | *36.116*                  |
| Gaussian Filter    | *35.177*                  |
| Refined Lee Filter | *14.0005*                  |
| Boxcar Filter      | *29.559*                  |

### Classification Outputs
- **Unsupervised Learning**: Segmented image using K-Means clustering.
- **Supervised Learning**:
  - SVM, Decision Tree, and Random Forest classifiers yielded visually distinct segmentation for urban, vegetation, and water classes.

## Requirements

- Python 3.8 or later
- Required libraries:
  - `GDAL`
  - `Pillow`
  - `numpy`
  - `scipy`
  - `scikit-learn`
  - `matplotlib`
  - `astropy`

## Future Enhancements

- Incorporate additional filters for noise reduction.
- Explore deep learning models for SAR image classification.
- Implement real-time processing for large datasets.
