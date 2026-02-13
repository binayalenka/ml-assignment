# Wine Quality Classification - ML Assignment 2

## Problem Statement

Wine quality assessment is traditionally performed by human experts through sensory analysis, which can be subjective and time-consuming. This project aims to develop machine learning models that can predict wine quality based on physicochemical properties, providing an objective and efficient alternative to manual assessment.

**Business Value:**
- Automated quality control in wine production
- Cost reduction by minimizing manual testing
- Consistent and objective quality assessment
- Early detection of quality issues in production

## Dataset Description

**Source:** UCI Machine Learning Repository - Wine Quality Dataset  
**Citation:** P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. Decision Support Systems, Elsevier, 47(4):547-553, 2009.

**Dataset Statistics:**
- **Total Samples:** 6,497 (1,599 red wine + 4,898 white wine)
- **Features:** 12 (11 physicochemical properties + 1 wine type indicator)
- **Target Variable:** Quality (score between 3 and 9)
- **Classification Type:** Multi-class classification (7 classes)
- **Train-Test Split:** 80-20 (5,197 training samples, 1,300 test samples)

**Feature Descriptions:**

| Feature | Description | Unit |
|---------|-------------|------|
| fixed acidity | Non-volatile acids (tartaric acid) | g/dm³ |
| volatile acidity | Acetic acid content (vinegar taste) | g/dm³ |
| citric acid | Adds freshness and flavor | g/dm³ |
| residual sugar | Sugar remaining after fermentation | g/dm³ |
| chlorides | Salt content | g/dm³ |
| free sulfur dioxide | Free form of SO₂ (prevents microbial growth) | mg/dm³ |
| total sulfur dioxide | Total SO₂ (free + bound forms) | mg/dm³ |
| density | Density of wine | g/cm³ |
| pH | Acidity level (0-14 scale) | - |
| sulphates | Wine additive (antimicrobial and antioxidant) | g/dm³ |
| alcohol | Alcohol content | % vol |
| wine_type | Type of wine (0=Red, 1=White) | - |

**Target Variable:**
- **Quality:** Score between 3-9 (higher is better)
- **Distribution:** Most wines rated 5-6 (normal quality), fewer at extremes

## Models Used

### Performance Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.5338 | 0.7210 | 0.5380 | 0.5338 | 0.4932 | 0.2513 |
| Decision Tree | 0.5623 | 0.7119 | 0.5554 | 0.5623 | 0.5554 | 0.3369 |
| K-Nearest Neighbors | 0.5585 | 0.7384 | 0.5441 | 0.5585 | 0.5489 | 0.3259 |
| Naive Bayes | 0.3215 | 0.5966 | 0.4207 | 0.3215 | 0.3621 | 0.0996 |
| Random Forest | **0.6915** | **0.8567** | **0.6960** | **0.6915** | **0.6786** | **0.5247** |
| XGBoost | 0.6531 | 0.8370 | 0.6458 | 0.6531 | 0.6449 | 0.4706 |

**Note:** Bold values indicate the best performance for each metric.

### Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| **Logistic Regression** | Moderate performance with 53.38% accuracy. The linear nature of the model struggles with the complex, non-linear relationships in wine quality. However, it achieves a decent AUC of 0.721, indicating reasonable class separation. The low F1 score (0.493) suggests difficulty in balancing precision and recall across all quality classes. Best suited as a baseline model. |
| **Decision Tree** | Shows improved performance over Logistic Regression with 56.23% accuracy. The tree structure captures non-linear patterns better. MCC of 0.337 indicates moderate correlation between predictions and true labels. However, limited max depth (10) prevents overfitting but may underfit complex patterns. Good interpretability makes it useful for understanding feature importance. |
| **K-Nearest Neighbors** | Achieves 55.85% accuracy with the best AUC among non-ensemble models (0.738). The instance-based learning approach works well for local patterns in the feature space. Performance depends heavily on the choice of k=5 neighbors. Computationally expensive for large datasets but provides smooth decision boundaries. Sensitive to feature scaling, which was properly addressed. |
| **Naive Bayes** | Poorest performance with only 32.15% accuracy and MCC of 0.100. The assumption of feature independence is strongly violated in this dataset, as physicochemical properties are highly correlated (e.g., density correlates with alcohol and sugar). The model's probabilistic nature makes it fast but inaccurate for this problem. Not recommended for wine quality prediction. |
| **Random Forest** | **Best overall performer** with 69.15% accuracy and highest scores across all metrics. The ensemble of 100 trees effectively captures complex interactions between features. Excellent AUC of 0.857 shows strong discriminative power. MCC of 0.525 indicates good correlation with true quality. Robust to outliers and handles class imbalance well. **Recommended for deployment.** |
| **XGBoost** | Second-best performer with 65.31% accuracy. The gradient boosting approach provides strong predictive power with AUC of 0.837. Slightly lower performance than Random Forest may be due to hyperparameter tuning needs. Excellent at handling imbalanced classes and missing values. Fast training and prediction. Good choice for production with proper tuning. |

### Key Insights

1. **Ensemble Methods Dominate:** Random Forest and XGBoost significantly outperform individual models, demonstrating the power of ensemble learning for this complex classification task.

2. **Feature Independence Assumption Fails:** Naive Bayes performs poorly because wine quality features are highly correlated, violating the independence assumption.

3. **Non-linearity Matters:** Tree-based and ensemble models outperform linear models, indicating that wine quality depends on complex, non-linear interactions between physicochemical properties.

4. **Class Imbalance Challenge:** The moderate accuracy scores (best: 69.15%) reflect the difficulty of predicting rare quality classes (3, 9) which have very few samples.

5. **Recommendation:** **Random Forest** is the best model for deployment, offering the highest accuracy, robustness, and interpretability through feature importance analysis.

## Streamlit Web Application

### Features

The deployed Streamlit app includes all required features:

1. ✅ **Dataset Upload Option** - Upload CSV files for prediction
2. ✅ **Model Selection Dropdown** - Choose from 6 trained models
3. ✅ **Evaluation Metrics Display** - View all 6 metrics for selected model
4. ✅ **Confusion Matrix & Classification Report** - Detailed performance analysis

### Additional Features

- **Interactive Visualizations** - Plotly charts for better insights
- **Model Comparison Tab** - Compare all models side-by-side
- **Prediction Tab** - Make predictions on uploaded data
- **Download Results** - Export predictions as CSV
- **Responsive Design** - Works on desktop and mobile

### Local Installation

```bash
# Clone the repository
git clone https://github.com/binayalenka/ml-assignment.git
cd ml-assignment

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Usage Instructions

1. **Select a Model:** Use the sidebar dropdown to choose one of the 6 models
2. **View Metrics:** Navigate to "Model Evaluation" tab to see performance metrics
3. **Upload Data:** Go to "Make Predictions" tab and upload a CSV file
4. **Compare Models:** Check "Model Comparison" tab to see all models side-by-side

## Project Structure

```
ML-Assignement/
├── app.py                          # Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── models/
│   ├── train_models.py            # Model training script
│   ├── logistic_regression.pkl    # Trained Logistic Regression
│   ├── decision_tree.pkl          # Trained Decision Tree
│   ├── k_nearest_neighbors.pkl    # Trained KNN
│   ├── naive_bayes.pkl            # Trained Naive Bayes
│   ├── random_forest.pkl          # Trained Random Forest
│   ├── xgboost.pkl                # Trained XGBoost
│   ├── scaler.pkl                 # Feature scaler
│   ├── label_encoder.pkl          # Label encoder
│   ├── metrics_summary.json       # All metrics in JSON
│   └── metrics_comparison.csv     # Metrics comparison table
└── data/
    ├── winequality-red.csv        # Red wine dataset
    ├── winequality-white.csv      # White wine dataset
    ├── wine_quality.csv           # Combined dataset
    └── test_sample.csv            # Test sample (100 rows)
```

## Deployment Links

- **GitHub Repository:** https://github.com/binayalenka/ml-assignment.git
- **Live Streamlit App:** https://ml-assignmentgit-dixrxvmeqmtux3aqggb43j.streamlit.app/

## Technical Details

**Libraries Used:**
- `scikit-learn` - Machine learning models and metrics
- `xgboost` - Gradient boosting classifier
- `streamlit` - Web application framework
- `plotly` - Interactive visualizations
- `pandas` - Data manipulation
- `numpy` - Numerical computations

**Model Training:**
- All models trained on 80% of data (5,197 samples)
- Tested on 20% of data (1,300 samples)
- Features scaled using StandardScaler
- Labels encoded for XGBoost compatibility
- Stratified split to maintain class distribution

**Evaluation Metrics:**
- **Accuracy:** Overall correctness
- **AUC:** Area Under ROC Curve (One-vs-Rest for multi-class)
- **Precision:** Weighted average across classes
- **Recall:** Weighted average across classes
- **F1 Score:** Harmonic mean of precision and recall
- **MCC:** Matthews Correlation Coefficient

## Assignment Compliance

This project fulfills all requirements of ML Assignment 2:

- ✅ Dataset with 12+ features and 500+ instances
- ✅ 6 classification models implemented
- ✅ 6 evaluation metrics calculated for each model
- ✅ Streamlit app with 4 required features
- ✅ Comprehensive README with all required sections
- ✅ GitHub repository with proper structure
- ✅ requirements.txt for deployment
- ✅ Ready for Streamlit Community Cloud deployment

## Author

**M.Tech (AIML/DSE) Student Binaya**  
Work Integrated Learning Programmes Division  
BITS Pilani

---

*This project was developed as part of Machine Learning Assignment 2 (15 marks)*  
*Submission Deadline: 15-Feb-2026*
