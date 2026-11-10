# Real-Time Chronic Kidney Disease Prediction Using Deep Learning: A TabNet-Based Explainable Screening System

## Overview

This project presents an AI-powered Chronic Kidney Disease (CKD) screening system using **TabNet**, a deep-learning architecture specifically designed for tabular data.

The system analyses clinical attributes, processes missing and inconsistent values, and predicts whether a given patient record indicates CKD or not. It also provides model-based explanations to help users understand the features influencing individual predictions.

The application integrates a **FastAPI backend**, **Streamlit frontend**, and **SQLite database** to provide an interactive prediction and monitoring experience.

> This project is developed for educational and research purposes. It is not intended to replace professional medical diagnosis or clinical decision-making.

## Key Features

- **TabNet-Based Prediction:** Uses deep learning for binary CKD classification.
- **Data Preprocessing:** Handles missing values, categorical encoding, standardisation and numerical scaling.
- **Class Imbalance Handling:** Uses SMOTE on the training dataset.
- **Explainable AI:** Provides global feature importance and individual prediction explanations using TabNet attention-based attribution.
- **Real-Time Prediction:** Accepts clinical attributes and returns predictions through a FastAPI service.
- **Risk-Level Presentation:** Displays predicted classification, confidence-related score and risk category.
- **Prediction History:** Stores prediction records and timestamps using SQLite.
- **Interactive Dashboard:** Streamlit frontend for entering clinical measurements and visualising results.

## Technology Stack

| Component | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TabNet, PyTorch |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, Imbalanced-learn |
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Database | SQLite |
| Data Visualisation | Matplotlib |
| API Communication | Requests |

## Dataset

The project uses the **UCI Chronic Kidney Disease dataset**.

- Total records: 400
- Clinical attributes: 24
- Classification: Binary (CKD / Not CKD)

The dataset contains clinical measurements and indicators such as:

- Age and Blood Pressure
- Blood Glucose
- Blood Urea
- Serum Creatinine
- Sodium and Potassium
- Haemoglobin
- Packed Cell Volume
- White Blood Cell Count
- Red Blood Cell Count
- Other clinical indicators

## System Architecture

The application follows a two-tier architecture.

### 1. Data Processing and Model Training

1. Load the CKD dataset.
2. Clean inconsistent categorical values.
3. Handle missing numerical and categorical values.
4. Encode categorical attributes.
5. Scale numerical features.
6. Perform an 80:20 stratified train-test split.
7. Apply SMOTE to the training data.
8. Train the TabNet classifier.
9. Evaluate the model using classification metrics.
10. Save the trained model and preprocessing artifacts.

### 2. Backend – FastAPI

The backend is responsible for:

- Loading the trained TabNet model.
- Applying the saved preprocessing transformations.
- Receiving clinical inputs through API requests.
- Generating CKD predictions.
- Computing confidence-related scores.
- Producing feature attribution information.
- Storing prediction history in SQLite.

### 3. Frontend – Streamlit

The frontend provides:

- Clinical input form.
- CKD prediction results.
- Risk-level presentation.
- Confidence score.
- Individual feature explanation charts.
- Global feature importance visualisation.
- Recent prediction history.

## Model Performance

The following results were reported on the held-out test dataset of 80 records.

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 98.75% |
| Precision | 96.77% |
| Recall | 100% |
| F1-Score | 98.36% |
| ROC-AUC | 1.00 |

The model correctly classified 79 out of 80 test records.

**Note:** These results represent the reported experimental split. The dataset is relatively small, and external validation is required before making claims about clinical generalisation.

## Explainable AI

One of the important aspects of this project is its focus on model explainability.

TabNet uses sequential attention mechanisms to select and emphasise relevant input features during different decision steps.

The system provides two explanation approaches:

**1. Global Feature Importance**

Identifies the features that contribute most to the model's predictions across the dataset.

**2. Local Feature Attribution**

Highlights the features that influenced an individual prediction, helping users inspect the model's decision process.

These explanations represent model behaviour and should not be interpreted as causal medical evidence.

## Project Structure

```text
CKD_Project/
│
├── artifacts/
│   ├── encoders.pkl
│   ├── feature_names.pkl
│   ├── scaler.pkl
│   └── tabnet_model.zip
│
├── backend/
│   ├── database.py
│   ├── main.py
│   ├── model_service.py
│   └── requirements.txt
│
├── data/
│   └── ckd.csv
│
├── processed/
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_test.npy
│   ├── X_train_balanced.npy
│   └── y_train_balanced.npy
│
├── results/
│   ├── sample_explanation.png
│   ├── tabnet_confusion_matrix.png
│   ├── tabnet_feature_importance.png
│   ├── tabnet_loss_curve.png
│   └── tabnet_metrics.csv
│
├── app.py
├── balance_data.py
├── explain_prediction.py
├── preprocess.py
├── train_tabnet.py
├── requirements.txt
└── .gitignore
```

## Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/aravindwork-prog/CKD-Prediction-TabNet.git
```

### 2. Navigate to the Project Directory

```bash
cd CKD-Prediction-TabNet
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

If the backend has additional dependencies, install those too:

```bash
pip install -r backend/requirements.txt
```

### 5. Start the FastAPI Backend

From the project root, the expected launch command is:

```bash
uvicorn backend.main:app --reload
```

If the FastAPI application instance has a different name or module path, adjust this command accordingly.

Backend documentation:

```text
http://localhost:8000/docs
```

### 6. Start the Streamlit Frontend

Open another terminal and execute:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Check backend and model status |
| `/predict` | POST | Generate CKD prediction |
| `/feature-importance` | GET | Retrieve global feature importance |
| `/history` | GET | Retrieve recent prediction records |

## Future Enhancements

- Validate the model using larger and independent clinical datasets.
- Compare TabNet with traditional machine-learning algorithms.
- Improve model calibration and prediction reliability.
- Enhance the explanation interface.
- Introduce secure authentication and access control.
- Explore cloud deployment and scalable database integration.

## Disclaimer

This application is an academic research prototype designed for preliminary CKD screening experiments.

It has not been clinically validated or approved for medical use. Predictions must not be used as a substitute for professional medical advice, laboratory investigations or diagnosis.

## Author

**Aravind S**

M.Tech – Software Engineering  
Vellore Institute of Technology (VIT), Vellore

GitHub: https://github.com/aravindwork-prog

---

⭐ If you find this project interesting, feel free to explore the repository and its implementation.