# 🛰️ AI-Based Satellite Image Classification for Land-Cover Detection

## 📌 Project Overview

This project uses deep learning to classify satellite images into 10 land-cover categories using the EuroSAT dataset. It includes a trained ResNet18 model and a Streamlit web application for image classification.

## 🧠 Model and Technologies

* **Model:** ResNet18
* **Dataset:** EuroSAT
* **Framework:** PyTorch
* **Programming Language:** Python
* **Web Application:** Streamlit

## 📊 Dataset

The project uses the EuroSAT dataset, containing satellite images across 10 land-cover classes.

* Total images: 27,000
* Training images: 18,900
* Validation images: 4,050
* Test images: 4,050
* Number of classes: 10

### Land-cover classes

AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake.

## 📈 Model Performance

* **Validation accuracy:** 91.95%
* **Training epochs:** 3
* **Training device:** CPU

*Note: This is the validation accuracy from the current training run. Test-set performance has not yet been reported. Model confidence scores are not guaranteed to indicate prediction correctness.*

## 🚀 Features

* Upload a satellite image (JPG or PNG).
* Predict its land-cover category.
* Display prediction confidence.
* Show the top predicted categories.

## 🛠️ Installation and Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/24A31A05JU/satellite-landcover-classification.git
   cd satellite-landcover-classification
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Download or prepare the EuroSAT dataset using the dataset script and place it in the expected data directory.

4. Run the Streamlit application:

   ```bash
   python -m streamlit run app.py
   ```

## 📂 Project Structure

```text
satellite-landcover-classification/
├── app.py
├── requirements.txt
├── src/
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── download_dataset.py
│   ├── split_dataset.py
│   └── test_model.py
├── data/
├── models/
└── outputs/
```

## 🔬 Future Improvements

* Evaluate performance on the held-out test set.
* Compare ResNet18 with other classification models.
* Improve performance through hyperparameter tuning and data augmentation.
* Add class-wise precision, recall, F1-score, and confusion matrix analysis.
* Explore applications in land-cover monitoring and remote sensing.

## 👩‍💻 Author

**Akanksha Talabattula**

GitHub: [24A31A05JU](https://github.com/24A31A05JU)

## ⚠️ Disclaimer

This project is a student research prototype for educational purposes. Predictions may be incorrect and should not be used as the sole basis for real-world land-management decisions.
