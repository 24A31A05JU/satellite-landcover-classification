# 🛰️ AI-Based Satellite Image Classification for Land-Cover Detection

## Project Overview
This project uses deep learning to classify satellite images into 10 land-cover categories using the EuroSAT dataset.

## Model and Technologies
- Model: ResNet18
- Dataset: EuroSAT
- Framework: PyTorch
- Programming Language: Python
- Web Application: Streamlit

## Dataset
EuroSAT contains 27,000 satellite images across 10 land-cover classes.

- Training images: 18,900
- Validation images: 4,050
- Test images: 4,050

## Preliminary Results
Validation accuracy: 91.95% after 3 epochs.

Note: This is a preliminary validation result, not final test accuracy. Performance on new images may vary.

## Features
- Upload a satellite image
- Predict its land-cover category
- Display model confidence
- Show top predictions

## How to Run
Install the required libraries:

pip install -r requirements.txt

Start the application:

streamlit run app.py

## Disclaimer
This is an academic research prototype. Predictions may be incorrect.
