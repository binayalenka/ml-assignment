"""
ML Assignment 2 - Wine Quality Classification
Streamlit Web Application for Model Demonstration
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Wine Quality Classifier",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #8B0000;
    }
    h2 {
        color: #A52A2A;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all trained models and preprocessing objects"""
    models = {}
    model_files = {
        'Logistic Regression': 'models/logistic_regression.pkl',
        'Decision Tree': 'models/decision_tree.pkl',
        'K-Nearest Neighbors': 'models/k_nearest_neighbors.pkl',
        'Naive Bayes': 'models/naive_bayes.pkl',
        'Random Forest': 'models/random_forest.pkl',
        'XGBoost': 'models/xgboost.pkl'
    }
    
    for name, path in model_files.items():
        with open(path, 'rb') as f:
            models[name] = pickle.load(f)
    
    # Load scaler and label encoder
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load metrics
    with open('models/metrics_summary.json', 'r') as f:
        metrics = json.load(f)
    
    return models, scaler, label_encoder, metrics

@st.cache_data
def load_test_data():
    """Load test sample data"""
    return pd.read_csv('data/test_sample.csv')

def preprocess_data(df, scaler):
    """Preprocess uploaded data"""
    # Remove quality column if present
    if 'quality' in df.columns:
        X = df.drop('quality', axis=1)
        y = df['quality']
    else:
        X = df
        y = None
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    return X_scaled, y

def plot_confusion_matrix(y_true, y_pred, label_encoder):
    """Create confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Get unique classes actually present in the data
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    class_labels = label_encoder.inverse_transform(unique_classes)
    
    fig = px.imshow(cm,
                    labels=dict(x="Predicted Quality", y="True Quality", color="Count"),
                    x=class_labels,
                    y=class_labels,
                    color_continuous_scale='RdYlGn',
                    text_auto=True,
                    aspect="auto")
    
    fig.update_layout(
        title="Confusion Matrix",
        width=600,
        height=500
    )
    
    return fig

def plot_metrics_comparison(metrics):
    """Create bar chart comparing all models across all metrics"""
    df_metrics = pd.DataFrame(metrics).T
    
    fig = go.Figure()
    
    for metric in df_metrics.columns:
        fig.add_trace(go.Bar(
            name=metric,
            x=df_metrics.index,
            y=df_metrics[metric],
            text=df_metrics[metric].round(3),
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score",
        barmode='group',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def main():
    # Header
    st.title("🍷 Wine Quality Classification")
    st.markdown("### ML Assignment 2 - Multi-Class Classification with 6 ML Models")
    st.markdown("---")
    
    # Load models and data
    with st.spinner("Loading models..."):
        models, scaler, label_encoder, metrics = load_models()
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection dropdown [REQUIRED FEATURE 2]
    st.sidebar.subheader("Select Model")
    model_name = st.sidebar.selectbox(
        "Choose a classification model:",
        list(models.keys()),
        help="Select one of the 6 trained models"
    )
    
    # Model description
    model_descriptions = {
        'Logistic Regression': 'Linear model for multi-class classification using One-vs-Rest strategy',
        'Decision Tree': 'Tree-based model with max depth of 10 for interpretable decisions',
        'K-Nearest Neighbors': 'Instance-based learning with k=5 neighbors',
        'Naive Bayes': 'Probabilistic classifier assuming Gaussian distribution of features',
        'Random Forest': 'Ensemble of 100 decision trees for robust predictions',
        'XGBoost': 'Gradient boosting ensemble with 100 estimators for high accuracy'
    }
    
    st.sidebar.info(f"**{model_name}**\n\n{model_descriptions[model_name]}")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📊 Model Evaluation", "🔮 Make Predictions", "📈 Model Comparison"])
    
    # Tab 1: Model Evaluation
    with tab1:
        st.header(f"Evaluation Metrics - {model_name}")
        
        # Display metrics [REQUIRED FEATURE 3]
        st.subheader("Performance Metrics")
        model_metrics = metrics[model_name]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{model_metrics['Accuracy']:.4f}")
            st.metric("AUC Score", f"{model_metrics['AUC']:.4f}")
        with col2:
            st.metric("Precision", f"{model_metrics['Precision']:.4f}")
            st.metric("Recall", f"{model_metrics['Recall']:.4f}")
        with col3:
            st.metric("F1 Score", f"{model_metrics['F1']:.4f}")
            st.metric("MCC Score", f"{model_metrics['MCC']:.4f}")
        
        st.markdown("---")
        
        # Confusion Matrix and Classification Report [REQUIRED FEATURE 4]
        st.subheader("Detailed Analysis")
        
        # Load test data for confusion matrix
        test_data = load_test_data()
        X_test, y_test = preprocess_data(test_data, scaler)
        
        # Get predictions
        model = models[model_name]
        y_pred = model.predict(X_test)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**Confusion Matrix**")
            fig_cm = plot_confusion_matrix(
                label_encoder.transform(y_test),
                y_pred,
                label_encoder
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            st.markdown("**Classification Report**")
            # Generate classification report
            # Get unique classes in predictions
            unique_classes = np.unique(np.concatenate([
                label_encoder.transform(y_test),
                y_pred
            ]))
            target_names = [str(label_encoder.inverse_transform([c])[0]) for c in unique_classes]
            
            report = classification_report(
                label_encoder.transform(y_test),
                y_pred,
                labels=unique_classes,
                target_names=target_names,
                output_dict=True,
                zero_division=0
            )
            
            # Convert to DataFrame for better display
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(
                report_df.style.format("{:.3f}"),
                use_container_width=True
            )
    
    # Tab 2: Make Predictions
    with tab2:
        st.header("Make Predictions on Your Data")
        
        # Dataset upload option [REQUIRED FEATURE 1]
        st.subheader("Upload Dataset (CSV)")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with wine features (without quality column for prediction)"
        )
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                df_upload = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded successfully! Shape: {df_upload.shape}")
                
                # Show data preview
                with st.expander("📋 View Data Preview"):
                    st.dataframe(df_upload.head(10))
                
                # Preprocess and predict
                X_upload, y_true = preprocess_data(df_upload, scaler)
                
                # Make predictions
                model = models[model_name]
                y_pred_encoded = model.predict(X_upload)
                y_pred_proba = model.predict_proba(X_upload)
                
                # Decode predictions
                y_pred = label_encoder.inverse_transform(y_pred_encoded)
                
                # Add predictions to dataframe
                df_results = df_upload.copy()
                df_results['Predicted Quality'] = y_pred
                df_results['Confidence'] = y_pred_proba.max(axis=1)
                
                if y_true is not None:
                    df_results['True Quality'] = y_true
                    df_results['Correct'] = df_results['True Quality'] == df_results['Predicted Quality']
                
                st.subheader("Prediction Results")
                st.dataframe(df_results, use_container_width=True)
                
                # Download results
                csv = df_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )
                
                # Show prediction distribution
                st.subheader("Prediction Distribution")
                fig_dist = px.histogram(
                    df_results,
                    x='Predicted Quality',
                    title="Distribution of Predicted Wine Quality",
                    color_discrete_sequence=['#8B0000']
                )
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # If true labels available, show accuracy
                if y_true is not None:
                    accuracy = (df_results['Correct'].sum() / len(df_results)) * 100
                    st.metric("Prediction Accuracy on Uploaded Data", f"{accuracy:.2f}%")
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("Please ensure your CSV has the correct format with wine quality features.")
        
        else:
            st.info("👆 Upload a CSV file to make predictions")
            
            # Show sample data format
            with st.expander("📝 Expected CSV Format"):
                st.write("Your CSV should contain the following columns:")
                sample_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                              'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                              'pH', 'sulphates', 'alcohol', 'wine_type']
                st.code(", ".join(sample_cols))
                
                # Load and show sample
                sample_data = load_test_data()
                st.write("Sample data (first 5 rows):")
                st.dataframe(sample_data.head())
    
    # Tab 3: Model Comparison
    with tab3:
        st.header("Compare All Models")
        
        # Metrics comparison table
        st.subheader("Performance Metrics Comparison")
        df_metrics = pd.DataFrame(metrics).T
        
        # Highlight best values
        st.dataframe(
            df_metrics.style.highlight_max(axis=0, color='lightgreen'),
            use_container_width=True
        )
        
        # Visual comparison
        st.subheader("Visual Comparison")
        fig_comparison = plot_metrics_comparison(metrics)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Best model recommendation
        st.subheader("🏆 Best Model Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            best_accuracy = df_metrics['Accuracy'].idxmax()
            st.success(f"**Best Accuracy:** {best_accuracy} ({df_metrics.loc[best_accuracy, 'Accuracy']:.4f})")
            
            best_f1 = df_metrics['F1'].idxmax()
            st.success(f"**Best F1 Score:** {best_f1} ({df_metrics.loc[best_f1, 'F1']:.4f})")
        
        with col2:
            best_auc = df_metrics['AUC'].idxmax()
            st.success(f"**Best AUC:** {best_auc} ({df_metrics.loc[best_auc, 'AUC']:.4f})")
            
            best_mcc = df_metrics['MCC'].idxmax()
            st.success(f"**Best MCC:** {best_mcc} ({df_metrics.loc[best_mcc, 'MCC']:.4f})")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>ML Assignment 2 - Wine Quality Classification</p>
        <p>Dataset: UCI Wine Quality | Models: 6 | Samples: 6,497</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
