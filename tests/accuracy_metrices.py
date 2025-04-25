import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from model_utils2 import train_model, detect_anomalies

@pytest.fixture
def metrics_data():
    """Generate data for testing multiple model metrics"""
    np.random.seed(42)
    n_samples = 500
    
    # Create features
    X = np.random.randn(n_samples, 6)
    
    # Create a target variable with some pattern
    # High values in first feature and low values in second feature indicate anomalies
    y = ((X[:, 0] > 0.5) & (X[:, 1] < -0.5)).astype(int)
    
    # Add some noise to make it more realistic
    noise_idx = np.random.choice(np.arange(n_samples), size=int(n_samples * 0.1), replace=False)
    y[noise_idx] = 1 - y[noise_idx]
    
    # Convert to DataFrame with named columns
    feature_cols = ['video_completion_rate', 'quiz_accuracy', 'avg_time_per_video',
                   'forum_activity', 'num_course_views', 'location_change']
    df = pd.DataFrame(X, columns=feature_cols)
    df['student_id'] = range(1, n_samples + 1)
    df['combined_anomaly'] = y
    
    return df

def test_compare_classifier_performance(metrics_data):
    """Test and compare the performance of different classifiers"""
    df = metrics_data
    features = ['video_completion_rate', 'quiz_accuracy', 'avg_time_per_video',
               'forum_activity', 'num_course_views', 'location_change']
    
    X = df[features]
    y = df['combined_anomaly']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Models to test
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    # Metrics to evaluate
    results = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC', 'CV Mean'])
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        # Cross validation
        cv_scores = cross_val_score(model, X, y, cv=5)
        cv_mean = cv_scores.mean()
        
        # Store results
        results = pd.concat([results, pd.DataFrame({
            'Model': [name],
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1': [f1],
            'ROC AUC': [roc_auc],
            'CV Mean': [cv_mean]
        })], ignore_index=True)
        
        # Print detailed classification report
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred))
    
    print("\nModel Comparison:")
    print(results)
    
    # Find best model
    best_model_idx = results['F1'].idxmax()
    best_model_name = results.loc[best_model_idx, 'Model']
    best_model_f1 = results.loc[best_model_idx, 'F1']
    
    print(f"\nBest model by F1 score: {best_model_name} with F1 = {best_model_f1:.4f}")
    
    # Basic assertions for model quality
    assert results['Accuracy'].min() >= 0.8, "At least one model has accuracy below 0.8"
    assert results['F1'].max() >= 0.4, "No model achieved F1 score of at least 0.4"
    
    # Write results to CSV for later reference
    import os
    output_dir = os.path.join(os.getcwd(), 'test_output')
    os.makedirs(output_dir, exist_ok=True)
    results.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)