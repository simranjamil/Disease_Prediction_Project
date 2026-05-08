# AI-Powered Disease Prediction and Expert Recommendation System

## Project Overview

This project is an AI-based Disease Prediction and Expert Recommendation System developed as part of the CT-361 Artificial Intelligence & Expert Systems course.

The system predicts possible diseases based on patient symptoms and basic health information using Machine Learning techniques. It also provides recommendations such as diet plans, medicines, and preventive measures through an integrated expert system.

The project is implemented as a web-based application using Flask.

---

# Features

- Disease prediction using Machine Learning
- Expert recommendations for treatment and prevention
- User-friendly web interface
- Flask-based backend integration
- Real-time prediction results
- Structured recommendation system

---

# Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML/CSS
- Pickle

---

# Dataset Information

The project uses two datasets:

## 1. Disease Symptoms Dataset
Contains:
- Symptoms
- Patient details
- Disease labels

Used for training the Machine Learning model.

## 2. Treatment & Prevention Dataset
Contains:
- Disease categories
- Recommended diets
- Suggested medicines
- Preventive measures

Used as the expert knowledge base.

---

# Input Features

The model uses the following features:

- Fever
- Cough
- Fatigue
- Difficulty Breathing
- Gender
- Blood Pressure
- Cholesterol Level

---

# Machine Learning Model

The following models were used:

- Decision Tree (baseline model)
- Random Forest Classifier (final model)

The Random Forest model was selected because of:
- Better generalization
- Improved stability
- Reduced overfitting

---

# Model Accuracy

The trained Random Forest model achieved approximately **57%–60% accuracy** during testing and evaluation.

Although the accuracy is moderate, the project successfully demonstrates:
- Disease prediction workflow
- Machine Learning integration
- Expert system recommendations
- Clinical decision support concept

This project is intended as a proof-of-concept academic system and not as a replacement for professional medical diagnosis.

---

# Project Workflow

1. User enters symptoms and patient information
2. Flask backend processes input data
3. Data is encoded into numerical format
4. Random Forest model predicts disease
5. Expert system fetches recommendations
6. Results are displayed on the web interface

---

# Project Structure

```bash
project-folder/
│
├── app.py
├── model.pkl
├── label_encoder.pkl
├── requirements.txt
├── datasets/
│   ├── disease_dataset.csv
│   └── treatment_dataset.csv
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── README.md
