"""
ML Assignment 2 - Model Training Pipeline
Trains 6 classification models on Wine Quality dataset and calculates evaluation metrics
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and combine red and white wine datasets"""
    print("Loading datasets...")
    
    # Load datasets
    red_wine = pd.read_csv('data/winequality-red.csv', sep=';')
    white_wine = pd.read_csv('data/winequality-white.csv', sep=';')
    
    # Add wine type column
    red_wine['wine_type'] = 0  # Red
    white_wine['wine_type'] = 1  # White
    
    # Combine datasets
    wine_data = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)
    
    print(f"Total samples: {len(wine_data)}")
    print(f"Features: {wine_data.shape[1] - 1}")
    print(f"Quality distribution:\n{wine_data['quality'].value_counts().sort_index()}")
    
    # Separate features and target
    X = wine_data.drop('quality', axis=1)
    y = wine_data['quality']
    
    return X, y, wine_data

def calculate_metrics(y_true, y_pred, y_pred_proba, model_name):
    """Calculate all 6 required evaluation metrics"""
    
    # Get unique classes
    classes = np.unique(y_true)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # For multi-class AUC, use One-vs-Rest approach
    try:
        # Binarize the labels for AUC calculation
        y_true_bin = label_binarize(y_true, classes=classes)
        
        # If binary classification (2 classes)
        if len(classes) == 2:
            auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            # Multi-class: use OvR strategy
            auc = roc_auc_score(y_true_bin, y_pred_proba, multi_class='ovr', average='weighted')
    except Exception as e:
        print(f"Warning: Could not calculate AUC for {model_name}: {e}")
        auc = 0.0
    
    # Weighted averages for multi-class
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    metrics = {
        'Accuracy': round(accuracy, 4),
        'AUC': round(auc, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1': round(f1, 4),
        'MCC': round(mcc, 4)
    }
    
    return metrics

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train all 6 models and calculate metrics"""
    
    results = {}
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr'),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, max_depth=6, 
                                 objective='multi:softmax', eval_metric='mlogloss')
    }
    
    print("\n" + "="*70)
    print("TRAINING AND EVALUATING MODELS")
    print("="*70)
    
    for model_name, model in models.items():
        print(f"\n{'='*70}")
        print(f"Training: {model_name}")
        print(f"{'='*70}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba, model_name)
        
        # Store results
        results[model_name] = {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        # Print metrics
        print(f"\nMetrics for {model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name:12s}: {value:.4f}")
        
        # Save model
        model_filename = model_name.lower().replace(' ', '_').replace('-', '_')
        with open(f'models/{model_filename}.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"  Model saved: models/{model_filename}.pkl")
    
    return results

def save_metrics_summary(results):
    """Save metrics summary to JSON file"""
    
    metrics_summary = {}
    for model_name, result in results.items():
        metrics_summary[model_name] = result['metrics']
    
    with open('models/metrics_summary.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    print("\n" + "="*70)
    print("METRICS SUMMARY SAVED")
    print("="*70)
    print("File: models/metrics_summary.json")
    
    # Print comparison table
    print("\n" + "="*70)
    print("MODEL COMPARISON TABLE")
    print("="*70)
    
    # Create DataFrame for better visualization
    df_metrics = pd.DataFrame(metrics_summary).T
    print(df_metrics.to_string())
    
    # Save as CSV for easy reference
    df_metrics.to_csv('models/metrics_comparison.csv')
    print("\nComparison table also saved as: models/metrics_comparison.csv")
    
    return df_metrics

def create_test_sample(wine_data):
    """Create a small test sample for Streamlit upload demo"""
    
    # Sample 100 random rows
    test_sample = wine_data.sample(n=100, random_state=42)
    test_sample.to_csv('data/test_sample.csv', index=False)
    
    print("\n" + "="*70)
    print("TEST SAMPLE CREATED")
    print("="*70)
    print("File: data/test_sample.csv (100 samples)")

def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("ML ASSIGNMENT 2 - MODEL TRAINING PIPELINE")
    print("="*70)
    
    # Load and prepare data
    X, y, wine_data = load_and_prepare_data()
    
    # Encode labels for XGBoost compatibility (convert quality scores to 0-indexed classes)
    print("\nEncoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Original classes: {label_encoder.classes_}")
    print(f"Encoded classes: {np.unique(y_encoded)}")
    
    # Split data
    print("\nSplitting data (80-20 train-test split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved: models/scaler.pkl")
    
    # Save label encoder
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("Label encoder saved: models/label_encoder.pkl")
    
    # Train and evaluate models
    results = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Save metrics summary
    df_metrics = save_metrics_summary(results)
    
    # Create test sample
    create_test_sample(wine_data)
    
    # Save full dataset
    wine_data.to_csv('data/wine_quality.csv', index=False)
    print(f"Full dataset saved: data/wine_quality.csv ({len(wine_data)} samples)")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nAll models trained and saved in 'models/' directory")
    print(f"Total models: {len(results)}")
    print(f"Best performing model (by Accuracy): {df_metrics['Accuracy'].idxmax()}")
    print(f"Best Accuracy: {df_metrics['Accuracy'].max():.4f}")

if __name__ == "__main__":
    main()
