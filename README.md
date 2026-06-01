# 🚀 AI-Driven Career Guidance & Roadmap System

A machine learning web application that predicts the optimal tech career track
based on a user's skill profile and generates a personalized learning roadmap.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Accuracy](https://img.shields.io/badge/Accuracy-90%25+-green)

---

## 🎯 Features

- **Skill Assessment** — Rate 20 technical & soft skills on a 0–10 scale
- **ML Prediction** — Random Forest classifier predicts your best-fit career
- **Top 3 Matches** — See confidence scores for your top career options
- **Personalized Roadmap** — Step-by-step learning path for your career
- **Skill Visualizer** — Bar chart of your current skill profile
- **Model Insights** — Confusion matrix & feature importance charts

---

## 🧠 ML Model Details

| Property | Detail |
|----------|--------|
| Algorithm | Random Forest Classifier |
| Trees | 200 estimators |
| Input Features | 20 skill ratings (0–10) |
| Output Classes | 8 career tracks |
| Validation | 5-Fold Stratified Cross Validation |
| Accuracy | 90%+ |

### Career Tracks Predicted
`Data Scientist` `Web Developer` `Mobile Developer` `DevOps Engineer`
`Cybersecurity Analyst` `AI/ML Engineer` `Cloud Architect` `Product Manager`

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.x |
| ML | Scikit-learn |
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

---

## 📁 Project Structure

career-guidance-system/
├── app.py                  # Streamlit dashboard
├── train_model.py          # ML model training script
├── generate_data.py        # Dataset generation script
├── data/
│   ├── career_dataset.csv  # Generated training data
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── models/
│   ├── career_model.pkl    # Trained model
│   └── label_encoder.pkl
├── requirements.txt
└── README.md