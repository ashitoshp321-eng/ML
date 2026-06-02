# Enterprise-Level Machine Learning Testing Platform

A complete, full-stack Machine Learning testing and execution playground. Built with a **Python Flask** backend and a responsive, interactive **Glassmorphism CSS** frontend. The platform supports model preprocessing, fitting, evaluation comparisons, AutoML suggestions, cloud-connected accuracy leaderboards, live inference testing, and ReportLab PDF reporting.

---

## Technical Architecture Stack

* **Backend Engine**: Python 3, Flask, Scikit-Learn, Pandas, NumPy, Joblib.
* **Database & Auth**: Firebase Firestore (with automatic local JSON database fallback).
* **Frontend Interface**: HTML5, CSS3 (Glassmorphic variables, dark/light theme switcher), JS (ES6), Bootstrap 5, Font Awesome, Plotly.js, Chart.js, AOS Animations.
* **Report Exporters**: ReportLab (PDF), python csv (CSV summaries).

---

## Core Features

1. **Flexible Data Sources**:
   * Preloaded sample datasets: `boston_housing.csv` (Regression), `iris_flower.csv` (Classification), and `mall_customers.csv` (Clustering).
   * Custom CSV uploading (stores files safely in `uploads/`).
2. **Interactive Exploratory Data Analysis (EDA)**:
   * View row/column counts, missing values, descriptive statistical summaries, and interactive dataset tables.
3. **Robust Data Preprocessing Pipeline**:
   * Clean missing numerical values (Mean/Median/Mode) and categorical columns (Mode).
   * Feature scaling selection (StandardScaler, MinMaxScaler, or raw values).
   * Categorical feature dummy hot-encoding and classification target auto-labeling.
   * Split training/test ratios (10% to 50%).
4. **Comprehensive Model Training Studio**:
   * Fit **7 regression algorithms** and **8 classification estimators**.
   * Tune hyperparameters on the fly via specialized dashboard form fields.
   * Save serialized model pipelines as `.pkl` in `saved_models/`.
5. **Interactive Model Diagnostics**:
   * Renders interactive Plotly visual graphs: Actual vs. Predicted curves, Residual scatter fits, Classification Confusion Matrices, ROC curves, and Feature Importances.
   * Downloads ReportLab PDF audit summaries and CSV metrics tables.
6. **Live Inference Predictor**:
   * Pre-compiles and builds a custom HTML prediction form matched to the model's active features (extracts categories as form dropdown selections).
   * Calculates classification confidence percentage.
   * Saves execution histories.
7. **Cross-Model Comparer**:
   * Ranks similar models (Regression or Classification) side-by-side using comparative tables and Plotly bar graphs.
8. **AutoML Intelligence Hub**:
   * Analyzes dataset metrics and recommends the best algorithm and parameters to train, detailing the reasoning behind the selection.
9. **Accuracy Leaderboards**:
   * Cloud-connected accuracy board ranking top-performing configurations.

---

## Installation & Setup

### 1. Clone the codebase and enter directory
```bash
cd c:\Users\HP\OneDrive\Desktop\ML
```

### 2. Install dependencies
Ensure you have Python installed, then run:
```bash
python -m pip install -r requirements.txt
```

### 3. Database Selection (Firebase or Fallback)

This platform features a dual-mode database helper:
* **Real Firebase**: Download your private key JSON file from the Firebase console, rename it to `firebase-credentials.json`, and place it in the project root folder.
* **Simulated Local Fallback**: If the credential file is missing, the system automatically falls back to storing data inside a local JSON file (`database/local_db.json`). This ensures the app runs immediately without any configuration required!

### 4. Running the Development Server
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your web browser.

---

## Directory Structure

```
project/
├── app.py                      # Flask main entrypoint
├── config.py                   # Configuration settings (Firebase config, upload folder, keys)
├── requirements.txt            # Dependency list (includes firebase-admin)
├── firebase-credentials.json   # User's Firebase service account key (ignored if not present)
├── database/
│   └── local_db.json           # Local JSON fallback DB (if Firebase key is missing)
├── uploads/                    # User uploaded CSVs
├── reports/                    # Generated PDFs & CSVs
├── saved_models/               # Serialized .pkl files
├── datasets/                   # Sample datasets preinstalled
│   ├── boston_housing.csv
│   ├── iris_flower.csv
│   └── mall_customers.csv
├── static/
│   ├── css/
│   │   └── style.css           # Custom Glassmorphism, animations, dark mode variables
│   ├── js/
│   │   └── main.js            # Frontend interactions, async uploads, training requests, charts
│   ├── images/
│   └── plots/                  # Static matplotlib/seaborn plots saved for PDF generation
├── templates/
│   ├── layouts/
│   │   └── base.html           # Main skeleton with responsive sidebar, navbar, dark mode toggler
│   ├── auth/
│   │   ├── login.html          # Login card
│   │   └── register.html       # Register card
│   ├── models/
│   │   └── index.html          # Models list (Supervised/Unsupervised cards)
│   ├── dashboard/
│   │   ├── regression.html     # Regression training & evaluation dashboard
│   │   ├── classification.html # Classification training & evaluation dashboard
│   │   ├── clustering.html     # Kmeans, Hierarchical, DBSCAN clustering dashboard
│   │   ├── dimensionality.html # PCA, t-SNE dashboard
│   │   ├── comparison.html     # Model comparison dashboard
│   │   ├── automl.html         # Auto ML recommendations page
│   │   └── leaderboard.html    # Leaderboard of all trained models
│   ├── index.html              # Landing home page
│   ├── about.html              # Explanatory & educational about page
│   └── contact.html            # Contact us page with interactive form
├── utils/
│   ├── firebase_db.py          # Firebase Firestore interaction and Local JSON fallback wrapper
│   ├── preprocessing.py        # Imputation, scaling, encoding, train/test split utilities
│   ├── visualization.py        # Matplotlib/Seaborn image generation + Plotly JSON exporter
│   ├── report_generator.py     # PDF & CSV reporting utilities using ReportLab
│   ├── model_trainer.py        # Model training logic, saving/loading joblib files
│   └── prediction.py           # Inference generation logic
└── README.md
```
