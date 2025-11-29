# Housing Price Prediction - MLOps Pipeline

A complete MLOps pipeline for predicting housing prices using the California Housing dataset. This project demonstrates end-to-end machine learning operations including data versioning with DVC, experiment tracking with MLflow, and continuous integration with GitHub Actions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [CI/CD Pipeline](#cicd-pipeline)
- [MLflow Tracking](#mlflow-tracking)
- [DVC Data Versioning](#dvc-data-versioning)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

---

## Project Overview

### Problem Statement
This project implements a **regression model** to predict median house values in California based on various features such as median income, house age, average rooms, and geographic location.

### ML Problem Type
- **Task**: Regression
- **Target Variable**: Median house value (in $100,000s)
- **Features**: 8 numerical features including median income, house age, average rooms, average bedrooms, population, average occupancy, latitude, and longitude

### Model
- **Algorithm**: Random Forest Regressor
- **Framework**: scikit-learn
- **Hyperparameters**: 
  - n_estimators: 100 (default)
  - max_depth: 10 (default)
  - random_state: 42

### Performance Metrics
The model is evaluated using:
- **R² Score**: Measures the proportion of variance explained
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error

---

##  Project Structure

```
mlops-mlflow-assignment/
│
├── .github/
│   └── workflows/
│       └── mlflow-pipeline.yml      # GitHub Actions CI/CD workflow
│
├── data/
│   |── raw_data.csv            # Raw dataset (DVC tracked)
│   └── raw_data.csv.dvc            # DVC metadata file
│
├── artifacts/
│   ├── scaler.pkl                  # Fitted StandardScaler
│   ├── feature_importance.csv      # Feature importance rankings
│   └── evaluation_metrics.json     # Model evaluation metrics
│
│ 
├── mlflow_pipeline.py              # Main pipeline script
├── MLproject                       # MLflow project configuration
├── python_env.yaml                 # Python environment specification
├── requirements.txt                # Python dependencies
├── .dvc/                           # DVC configuration
├── .dvcignore                      # DVC ignore patterns
├── .gitignore                      # Git ignore patterns
└── README.md                       # This file
```

---

##  Technologies Used

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Programming Language | 3.9+ |
| **MLflow** | Experiment Tracking & Model Registry | 2.x |
| **DVC** | Data Version Control | 3.x |
| **scikit-learn** | Machine Learning Library | 1.3+ |
| **pandas** | Data Manipulation | 2.x |
| **NumPy** | Numerical Computing | 1.24+ |
| **GitHub Actions** | CI/CD Pipeline | - |
| **Google Drive** | Remote Storage for DVC | - |

---

##  Setup Instructions

### Prerequisites

- Python 3.9 or higher
- Git installed
- Google Drive account (for DVC remote storage)
- GitHub account

### 1. Clone the Repository

```bash
git clone https://github.com/mehru4321/mlops-mlflow-assignment.git
cd YOUR_REPO_NAME
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set Up DVC Remote Storage (Google Drive)

#### Step 4.1: Initialize DVC (if not already initialized)

```bash
dvc init
```

#### Step 4.2: Configure Google Drive Remote

```bash
# Add Google Drive as remote storage
dvc remote add -d gdrive gdrive://YOUR_GDRIVE_FOLDER_ID

# Authenticate with Google Drive
dvc remote modify gdrive gdrive_acknowledge_abuse true
```

**To get your Google Drive Folder ID:**
1. Create a folder in Google Drive (e.g., "mlops-data")
2. Open the folder
3. Copy the ID from the URL: `https://drive.google.com/drive/folders/FOLDER_ID`
4. Replace `YOUR_GDRIVE_FOLDER_ID` with this ID

#### Step 4.3: Pull Data from DVC

```bash
# Pull the dataset from remote storage
dvc pull

# Verify data is downloaded
ls data/raw/raw_data.csv
```

### 5. Set Up MLflow Tracking

MLflow uses a local file-based backend by default. No additional setup required!

**Optional: Use SQLite Backend (Recommended for Production)**

```bash
# Create SQLite database for tracking
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

---

##  Pipeline Walkthrough

### Pipeline Architecture

The pipeline consists of **4 main stages**:

```
┌─────────────────┐
│  1. Data        │
│  Extraction     │──┐
└─────────────────┘  │
                     │
┌─────────────────┐  │
│  2. Data        │  │
│  Preprocessing  │◄─┘
└─────────────────┘  │
                     │
┌─────────────────┐  │
│  3. Model       │  │
│  Training       │◄─┘
└─────────────────┘  │
                     │
┌─────────────────┐  │
│  4. Model       │  │
│  Evaluation     │◄─┘
└─────────────────┘
```

### Stage 1: Data Extraction
- Loads California Housing dataset
- Saves raw data to `data/raw/raw_data.csv`
- Logs data dimensions and feature names to MLflow

### Stage 2: Data Preprocessing
- Splits data into train/test sets (80/20)
- Standardizes features using StandardScaler
- Saves scaler for future predictions
- Logs preprocessing parameters to MLflow

### Stage 3: Model Training
- Trains Random Forest Regressor
- Calculates feature importance
- Logs model parameters and training metrics
- Registers model in MLflow Model Registry

### Stage 4: Model Evaluation
- Evaluates model on test set
- Calculates R², RMSE, MAE, MAPE
- Logs all evaluation metrics to MLflow
- Saves metrics to JSON file

---

##  Running the Pipeline

### Method 1: Direct Python Execution (Recommended)

```bash
# Run with default parameters
python mlflow_pipeline.py

# Run with custom parameters
python mlflow_pipeline.py \
    --data-path data/raw/raw_data.csv \
    --test-size 0.2 \
    --n-estimators 150 \
    --max-depth 15 \
    --random-state 42
```

### Method 2: Using MLflow Projects

```bash
# Run with default parameters
mlflow run .

# Run with custom parameters
mlflow run . -P n_estimators=150 -P max_depth=15
```

### Expected Output

```
======================================================================
HOUSING PRICE PREDICTION PIPELINE
MLflow-Based Pipeline Execution
======================================================================

Started at: 2025-11-29 19:40:30

Pipeline Parameters:
  data_path: data/raw/raw_data.csv
  test_size: 0.2
  n_estimators: 100
  max_depth: 10
  random_state: 42

======================================================================
STEP 1: DATA EXTRACTION
======================================================================
✓ Data loaded: (20640, 9)

======================================================================
STEP 2: DATA PREPROCESSING
======================================================================
Training samples: 16,512
Test samples: 4,128
✓ Features standardized (StandardScaler)

======================================================================
STEP 3: MODEL TRAINING
======================================================================
✓ Model trained
  Training R²: 0.8719
  Training RMSE: 0.4138

======================================================================
STEP 4: MODEL EVALUATION
======================================================================
Test R² Score:    0.7739   (Higher is better, max=1.0)
Test RMSE:        0.5443   (Lower is better)
Test MAE:         0.3663   (Lower is better)
Test MAPE:        21.58%  (Lower is better)

======================================================================
PIPELINE COMPLETED SUCCESSFULLY!
======================================================================
```

---

##  MLflow Tracking

### View MLflow UI

```bash
# Start MLflow UI
mlflow ui --port 5000

# Open in browser
# http://localhost:5000
```

### What You Can Track

1. **Experiments**: View all pipeline runs
2. **Parameters**: Model hyperparameters, data split ratios
3. **Metrics**: R², RMSE, MAE, MAPE, feature importance
4. **Artifacts**: Trained models, scalers, feature importance CSVs
5. **Models**: Registered models in Model Registry

### MLflow UI Features

- **Compare Runs**: Compare metrics across different runs
- **Visualize Metrics**: Plot metric trends over time
- **Model Registry**: Version and stage models (Staging, Production)
- **Artifacts Browser**: Download saved models and artifacts

---

##  DVC Data Versioning

### Track New Data

```bash
# Track raw data with DVC
dvc add data/raw/raw_data.csv

# Commit DVC metadata to Git
git add data/raw/raw_data.csv.dvc .gitignore
git commit -m "Track raw data with DVC"
```

### Push Data to Remote

```bash
# Push data to Google Drive
dvc push

# Verify data is uploaded
dvc status -r gdrive
```

### Pull Data from Remote

```bash
# Pull data from Google Drive
dvc pull

# Pull specific file
dvc pull data/raw/raw_data.csv.dvc
```

### Switch Between Data Versions

```bash
# Checkout specific Git commit
git checkout <commit-hash>

# Pull corresponding data version
dvc checkout
```

---

##  CI/CD Pipeline

### GitHub Actions Workflow

The project uses **GitHub Actions** for continuous integration with 3 main stages:

#### **Stage 1: Environment Setup**
- Checkout code from repository
- Set up Python environment
- Install dependencies from `requirements.txt`
- Verify installations

#### **Stage 2: Pipeline Validation & Compilation**
- Validate MLflow project structure
- Check Python syntax
- Validate pipeline components
- Generate `pipeline_metadata.yaml`

#### **Stage 3: Pipeline Execution & Testing**
- Create required directories
- Run the complete pipeline
- Verify all outputs (data, artifacts, MLflow runs)
- Display evaluation metrics

### Trigger CI/CD

**Automatic Trigger:**
- Push to `main` branch
- Create pull request to `main` branch

**Manual Trigger:**
1. Go to GitHub repository
2. Click **Actions** tab
3. Select **"MLflow Pipeline CI"**
4. Click **"Run workflow"**

### View CI/CD Results

```
GitHub Repo → Actions tab → Click on workflow run
```

---

##  Results

### Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Test R²** | 0.7739 | Model explains 77.39% of variance |
| **Test RMSE** | 0.5443 | Average prediction error of $54,430 |
| **Test MAE** | 0.3663 | Average absolute error of $36,630 |
| **Test MAPE** | 21.58% | Average percentage error of 21.58% |

### Top 3 Most Important Features

1. **MedInc** (0.5938) - Median Income
2. **AveOccup** (0.1398) - Average Occupancy
3. **Latitude** (0.0766) - Geographic Latitude

---

##  Troubleshooting

### Common Issues

#### Issue 1: MLflow Not Found
```bash
# Solution: Install MLflow
pip install mlflow
```

#### Issue 2: Data File Not Found
```bash
# Solution: Pull data from DVC
dvc pull
```

#### Issue 3: Permission Denied (DVC)
```bash
# Solution: Authenticate with Google Drive
dvc remote modify gdrive gdrive_use_service_account false
dvc pull  # Will prompt for authentication
```

#### Issue 4: GitHub Actions Failing
```bash
# Solution: Check workflow logs in GitHub Actions tab
# Common fixes:
# - Update requirements.txt
# - Fix Python syntax errors
# - Verify file paths
```

#### Issue 5: Port Already in Use (MLflow UI)
```bash
# Solution: Use different port
mlflow ui --port 5001


## Acknowledgments

- California Housing dataset from scikit-learn
- MLflow for experiment tracking
- DVC for data version control
- GitHub Actions for CI/CD automation

---

## Project Status

 **Completed** - All MLOps components implemented and tested

Last Updated: November 29, 2025
