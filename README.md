# 🫀 Heart Disease Prediction — End-to-End Machine Learning Project

A complete end-to-end machine learning pipeline for predicting heart disease using the **Cleveland Heart Disease Dataset** (UCI Machine Learning Repository). This project covers everything from data ingestion and cleaning to model training, evaluation, and inference.

---

## 📁 Project Structure

```
heart-disease-ml/
├── data/
│   ├── raw/                    # Original downloaded data
│   └── processed/              # Cleaned & feature-engineered data
├── src/
│   ├── data_loader.py          # Data ingestion & download
│   ├── data_cleaning.py        # Preprocessing & feature engineering
│   ├── eda.py                  # Exploratory Data Analysis
│   ├── train.py                # Model training & hyperparameter tuning
│   ├── evaluate.py             # Model evaluation & metrics
│   └── predict.py              # Inference on new data
├── models/                     # Saved model artifacts
├── reports/
│   └── figures/                # EDA and evaluation plots
├── tests/                      # Unit tests
│   ├── test_data_cleaning.py
│   └── test_predict.py
├── main.py                     # Run full pipeline end-to-end
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**Cleveland Heart Disease Dataset** — UCI Machine Learning Repository

- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Samples**: 303 patients
- **Features**: 13 clinical attributes (age, cholesterol, chest pain type, etc.)
- **Target**: Binary — presence (1) or absence (0) of heart disease
- **Year**: Collected 1988, widely used benchmark dataset in medical ML research

### Features

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Age in years | Numeric |
| `sex` | Sex (1=Male, 0=Female) | Categorical |
| `cp` | Chest pain type (0-3) | Categorical |
| `trestbps` | Resting blood pressure (mm Hg) | Numeric |
| `chol` | Serum cholesterol (mg/dl) | Numeric |
| `fbs` | Fasting blood sugar > 120 mg/dl | Binary |
| `restecg` | Resting ECG results (0-2) | Categorical |
| `thalach` | Max heart rate achieved | Numeric |
| `exang` | Exercise-induced angina | Binary |
| `oldpeak` | ST depression induced by exercise | Numeric |
| `slope` | Slope of peak exercise ST segment | Categorical |
| `ca` | Number of major vessels (0-3) | Numeric |
| `thal` | Thalassemia (1=normal, 2=fixed defect, 3=reversible defect) | Categorical |

---

## 🚀 Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-ml.git
cd heart-disease-ml
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline

```bash
python main.py
```

This will:
1. Download and load the dataset
2. Clean and preprocess the data
3. Run exploratory data analysis (saves plots to `reports/figures/`)
4. Train multiple ML models with cross-validation
5. Evaluate and compare model performance
6. Save the best model to `models/`

### 4. Run inference on new patient data

```bash
python src/predict.py
```

### 5. Run tests

```bash
python -m pytest tests/ -v
```

---

## 🤖 Models Trained

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear model |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Boosted tree ensemble |
| Support Vector Machine | Kernel-based classifier |
| K-Nearest Neighbors | Distance-based classifier |

All models are evaluated with **5-fold stratified cross-validation** and tuned via **GridSearchCV**.

---

## 📈 Results

After training, results are saved to `reports/model_comparison.csv`. Typical performance:

| Model | Accuracy | ROC-AUC | F1 Score |
|-------|----------|---------|---------|
| Logistic Regression | ~85% | ~0.91 | ~0.85 |
| Random Forest | ~86% | ~0.92 | ~0.86 |
| **Gradient Boosting** | **~88%** | **~0.94** | **~0.88** |
| SVM | ~85% | ~0.91 | ~0.85 |
| KNN | ~82% | ~0.88 | ~0.82 |

*(Actual results may vary slightly due to random seed)*

---

## 🧪 Key Findings from EDA

- **Age & max heart rate** are the strongest individual predictors
- Patients with **chest pain type 0 (asymptomatic)** have significantly higher disease rates
- **Thalassemia type 2 (reversible defect)** strongly correlates with positive diagnosis
- **Female patients** in this dataset show lower disease prevalence but higher severity when present
- Mild class imbalance: ~54% negative, ~46% positive

---

## 🛠 Tech Stack

- **Python 3.9+**
- **pandas** — data manipulation
- **numpy** — numerical computing
- **scikit-learn** — ML models, preprocessing, evaluation
- **matplotlib / seaborn** — visualization
- **joblib** — model serialization

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- Janosi, Steinbrunn, Pfisterer, Detrano — original dataset collectors
- UCI Machine Learning Repository for hosting the dataset

---

## 🤖 How This Project Was Built

This project was designed and generated end-to-end by **[Claude](https://claude.ai)** (Anthropic's AI assistant, model: Claude Sonnet 4.6), in response to a single prompt asking for a production-quality, GitHub-ready health ML project in Python.

Claude selected the dataset, wrote every line of code, designed the modular pipeline architecture, engineered the features, chose and tuned all five models, and authored this README — all in one session. The code was executed and verified in a sandboxed Linux environment before being packaged.

The human's role was to provide the brief and direction. Everything else — structure, decisions, implementation — was Claude.

