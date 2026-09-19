# 🛰️ AI-Based Satellite Image Classification for Land-Cover Detection

A deep-learning project that classifies satellite images into different land-cover categories using the **EuroSAT dataset** and a **ResNet18** model.

## 📌 Project Overview

This project uses computer vision to identify land-cover types from satellite images. A user can upload an image through a Streamlit web application and receive the predicted land-cover class along with confidence scores.

## 🎯 Objectives

* Classify satellite images into 10 land-cover categories.
* Apply deep learning using a ResNet18 architecture.
* Evaluate model performance using accuracy, precision, recall, and F1-score.
* Build an interactive web application for image classification.

## 🗂️ Dataset

**EuroSAT Dataset**

* 27,000 satellite images
* 10 land-cover classes
* RGB satellite images

### Classes

AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake.

The dataset was divided into training, validation, and test sets:

| Dataset    |     Images |
| ---------- | ---------: |
| Training   |     18,900 |
| Validation |      4,050 |
| Testing    |      4,050 |
| **Total**  | **27,000** |

## 🧠 Model

**ResNet18 (pretrained)**

The pretrained feature layers are frozen, and a new classification layer is trained for the 10 EuroSAT classes.

## 📊 Preliminary Results

| Metric              | Result |
| ------------------- | -----: |
| Validation accuracy | 91.95% |
| Test accuracy       | 91.73% |

These are results from the current experiment and may vary with training settings or dataset splits.

## 🛠️ Technologies Used

* Python
* PyTorch
* Torchvision
* Pandas
* Scikit-learn
* Matplotlib and Seaborn
* Streamlit
* Pillow

## 🚀 Features

* Upload satellite images in JPG or PNG format.
* Predict the land-cover category.
* Display model confidence.
* Show the top three predictions.
* Evaluate model performance with a classification report and confusion matrix.

## 📁 Project Structure

```text
satellite-landcover-classification/
├── app.py
├── train_model.py
├── evaluate_model.py
├── class_distribution.py
├── download_dataset.py
├── inspect_dataset.py
├── plot_distribution.py
├── split_dataset.py
├── test_dataloader.py
├── test_model.py
├── view_image.py
├── requirements.txt
├── README.md
├── data/
├── models/
└── outputs/
    ├── class_distribution.png
    ├── classification_report.csv
    ├── confusion_matrix.png
    ├── train.csv
    ├── validation.csv
    └── test.csv
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/24A31A05JU/satelite-landcover-classification.git
cd satelite-landcover-classification
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Ensure the EuroSAT dataset is available in the expected `data/` directory and that the CSV split files are present in `outputs/`.

### 5. Train the model

```bash
python train_model.py
```

### 6. Evaluate the model

```bash
python evaluate_model.py
```

### 7. Run the web application

```bash
streamlit run app.py
```

Upload a satellite image in the browser to view its predicted land-cover category.

## 📈 Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

The evaluation report is saved as `outputs/classification_report.csv`, and the confusion matrix is saved as `outputs/confusion_matrix.png`.

## 🔬 Research Scope

This project serves as a prototype for exploring deep-learning-based land-cover classification from satellite imagery. Future work may include comparisons with other architectures, expanded evaluation, and testing on satellite imagery from different sources.

## ⚠️ Limitations

* The model is trained on the EuroSAT dataset and may not generalize to all satellite imagery.
* Predictions depend on image quality and similarity to the training data.
* Confidence scores do not guarantee that a prediction is correct.
* The current implementation is a research prototype, not an operational remote-sensing system.

## 👩‍💻 Author

**Akanksha Talabattula**

GitHub: [24A31A05JU](https://github.com/24A31A05JU)

---

*Developed as a student deep-learning project exploring satellite image classification and land-cover detection.*
