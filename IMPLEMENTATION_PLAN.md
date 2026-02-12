# ML Assignment 2 - Classification Models & Streamlit Deployment

## Overview

This assignment requires implementing 6 classification models on a chosen dataset, evaluating them with 6 metrics, creating a Streamlit web application, and deploying it to Streamlit Community Cloud.

**Dataset Choice**: Wine Quality Dataset from UCI/Kaggle
- **Features**: 12 (11 physicochemical + 1 quality target)
- **Instances**: ~4,900 (red + white wine combined)
- **Type**: Multi-class classification (quality scores 3-9)
- **Source**: https://archive.ics.uci.edu/ml/datasets/wine+quality

This dataset is ideal because:
- Meets minimum requirements (12 features, 500+ instances)
- Well-documented and widely used
- Real-world application (wine quality prediction)
- Balanced complexity for all 6 models

## User Review Required

> [!IMPORTANT]
> **Dataset Selection**: I've chosen the Wine Quality dataset. If you prefer a different dataset (e.g., Heart Disease, Customer Segmentation), please let me know before I proceed.

> [!IMPORTANT]
> **Binary vs Multi-class**: The Wine Quality dataset is multi-class (quality scores 3-9). I can convert it to binary classification (good/bad quality) if preferred.

## Proposed Changes

### Project Structure

```
ML-Assignement/
├── app.py                          # Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Comprehensive documentation
├── models/
│   ├── train_models.py            # Model training script
│   ├── logistic_regression.pkl    # Saved models
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
├── data/
│   ├── wine_quality.csv           # Full dataset
│   └── test_sample.csv            # Small test sample for Streamlit upload
└── notebooks/
    └── model_development.ipynb    # Development & analysis notebook
```

---

### Data Preparation & Model Training

#### [NEW] [train_models.py](file:///Users/kumarsankalp/Downloads/ML-Assignement/models/train_models.py)

**Purpose**: Complete ML pipeline for training all 6 models and calculating metrics

**Implementation**:
1. **Data Loading & Preprocessing**
   - Load Wine Quality dataset
   - Handle missing values (if any)
   - Feature scaling using StandardScaler
   - Train-test split (80-20)

2. **Model Implementation** (6 models):
   - Logistic Regression (multi-class with OvR strategy)
   - Decision Tree Classifier
   - K-Nearest Neighbors (k=5, optimized via cross-validation)
   - Naive Bayes (GaussianNB for continuous features)
   - Random Forest (n_estimators=100)
   - XGBoost (multi-class with softmax)

3. **Evaluation Metrics** (6 metrics per model):
   - Accuracy
   - AUC Score (using OvR for multi-class)
   - Precision (weighted average)
   - Recall (weighted average)
   - F1 Score (weighted average)
   - Matthews Correlation Coefficient (MCC)

4. **Model Persistence**
   - Save all trained models as .pkl files
   - Save scaler for consistent preprocessing
   - Export metrics to JSON for README table

---

### Streamlit Application

#### [NEW] [app.py](file:///Users/kumarsankalp/Downloads/ML-Assignement/app.py)

**Purpose**: Interactive web application for model demonstration

**Features** (as per assignment requirements):

1. **Dataset Upload Option** [1 mark]
   - File uploader for CSV files
   - Display uploaded data preview
   - Automatic preprocessing (scaling)
   - Handle test data only (due to Streamlit free tier limits)

2. **Model Selection Dropdown** [1 mark]
   - Dropdown with all 6 models
   - Load selected model dynamically
   - Display model description

3. **Evaluation Metrics Display** [1 mark]
   - Show all 6 metrics in formatted table
   - Visual comparison charts (bar plots)
   - Highlight best performing model per metric

4. **Confusion Matrix & Classification Report** [1 mark]
   - Interactive confusion matrix heatmap
   - Detailed classification report
   - Per-class precision, recall, F1 scores

**UI Design**:
- Clean, professional layout with sidebar navigation
- Color-coded metrics (green for high, red for low)
- Responsive design for mobile/desktop
- Loading indicators for model predictions

---

### Documentation

#### [NEW] [README.md](file:///Users/kumarsankalp/Downloads/ML-Assignement/README.md)

**Structure** (as per assignment requirements):

1. **Problem Statement**
   - Wine quality prediction based on physicochemical properties
   - Business value and applications

2. **Dataset Description** [1 mark]
   - Source and citation
   - Feature descriptions (all 12 features)
   - Target variable explanation
   - Dataset statistics

3. **Models Used** [6 marks - 1 per model]
   - Comparison table with all 6 metrics for all 6 models
   - Model-specific observations table

4. **Observations** [3 marks]
   - Performance analysis for each model
   - Strengths and weaknesses
   - Recommendations for deployment

5. **Deployment Instructions**
   - How to run locally
   - How to use the Streamlit app
   - GitHub and live app links

---

### Dependencies

#### [NEW] [requirements.txt](file:///Users/kumarsankalp/Downloads/ML-Assignement/requirements.txt)

```
streamlit==1.31.0
scikit-learn==1.4.0
xgboost==2.0.3
numpy==1.26.3
pandas==2.2.0
matplotlib==3.8.2
seaborn==0.13.1
plotly==5.18.0
```

**Note**: Specific versions to avoid deployment issues on Streamlit Cloud

---

### Development Notebook

#### [NEW] [model_development.ipynb](file:///Users/kumarsankalp/Downloads/ML-Assignement/notebooks/model_development.ipynb)

**Purpose**: Exploratory data analysis and model experimentation

**Contents**:
- EDA with visualizations
- Feature correlation analysis
- Hyperparameter tuning experiments
- Model comparison analysis
- Detailed metric calculations

## Verification Plan

### Automated Tests

1. **Model Training Verification**
   ```bash
   cd /Users/kumarsankalp/Downloads/ML-Assignement
   python models/train_models.py
   ```
   - Verify all 6 .pkl files created in `models/` directory
   - Check metrics JSON file generated
   - Confirm no errors during training

2. **Local Streamlit App Testing**
   ```bash
   cd /Users/kumarsankalp/Downloads/ML-Assignement
   streamlit run app.py
   ```
   - Test dataset upload with sample CSV
   - Verify all 6 models load correctly
   - Check metrics display properly
   - Confirm confusion matrix renders
   - Test all interactive features

3. **Requirements Validation**
   ```bash
   pip install -r requirements.txt
   ```
   - Ensure all dependencies install without conflicts
   - Test on clean virtual environment

### Manual Verification

1. **GitHub Repository Check**
   - Verify repository structure matches plan
   - Check all files committed and pushed
   - Confirm README.md renders correctly on GitHub
   - Validate requirements.txt completeness

2. **Streamlit Cloud Deployment**
   - Deploy to Streamlit Community Cloud
   - Test live app functionality
   - Verify app loads without errors
   - Test with sample data upload
   - Confirm all features work in production

3. **BITS Virtual Lab Execution**
   - Run complete pipeline on BITS Virtual Lab
   - Take screenshot showing successful execution
   - Verify timestamp visible in screenshot

4. **Submission PDF Creation**
   - Include GitHub repository link
   - Include live Streamlit app link
   - Include BITS Lab screenshot
   - Include complete README.md content
   - Verify all links are clickable

### Success Criteria

✅ All 6 models trained and saved successfully  
✅ All 6 metrics calculated for each model  
✅ Streamlit app runs locally without errors  
✅ App deployed successfully to Streamlit Cloud  
✅ All 4 required app features implemented  
✅ README.md complete with all required sections  
✅ GitHub repository properly structured  
✅ BITS Lab screenshot obtained  
✅ Submission PDF contains all required elements
