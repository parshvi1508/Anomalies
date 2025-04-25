import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

from model_utils2 import train_model, detect_anomalies

@pytest.fixture
def generate_synthetic_data():
    """Generate synthetic data with known patterns for testing"""
    np.random.seed(42)
    n_samples = 300
    
    # Create normal students (70%)
    normal_count = int(n_samples * 0.7)
    normal_students = pd.DataFrame({
        'student_id': range(1, normal_count + 1),
        'video_completion_rate': np.random.uniform(70, 100, normal_count),
        'quiz_accuracy': np.random.uniform(60, 100, normal_count),
        'avg_time_per_video': np.random.uniform(5, 25, normal_count),
        'forum_activity': np.random.randint(3, 10, normal_count),
        'num_course_views': np.random.randint(30, 100, normal_count),
        'location_change': np.random.randint(0, 3, normal_count)
    })
    
    # Create at-risk students (30%)
    at_risk_count = n_samples - normal_count
    at_risk_students = pd.DataFrame({
        'student_id': range(normal_count + 1, n_samples + 1),
        'video_completion_rate': np.random.uniform(10, 60, at_risk_count),
        'quiz_accuracy': np.random.uniform(10, 50, at_risk_count),
        'avg_time_per_video': np.random.uniform(30, 60, at_risk_count),
        'forum_activity': np.random.randint(0, 3, at_risk_count),
        'num_course_views': np.random.randint(5, 30, at_risk_count),
        'location_change': np.random.randint(4, 10, at_risk_count)
    })
    
    # Combine datasets
    df = pd.concat([normal_students, at_risk_students], ignore_index=True)
    
    # Shuffle rows
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def test_model_performance_metrics(generate_synthetic_data):
    """Test model performance using standard metrics"""
    df = generate_synthetic_data
    
    # Train model
    model, metrics = train_model(df)
    
    # Standard evaluation metrics should be available
    assert 'accuracy' in metrics, "Accuracy metric missing"
    assert 'classification_report' in metrics, "Classification report missing"
    assert 'confusion_matrix' in metrics, "Confusion matrix missing"
    
    # Check accuracy threshold
    assert metrics['accuracy'] >= 0.75, f"Model accuracy {metrics['accuracy']} below threshold of 0.75"
    
    # Extract and check precision and recall from classification report
    precision = metrics['classification_report']['1']['precision']
    recall = metrics['classification_report']['1']['recall']
    
    assert precision >= 0.7, f"Precision {precision} below threshold of 0.7"
    assert recall >= 0.7, f"Recall {recall} below threshold of 0.7"

def test_cross_validation_performance(generate_synthetic_data):
    """Test model with cross-validation"""
    df = generate_synthetic_data
    
    # Prepare data
    features = ['video_completion_rate', 'quiz_accuracy', 'avg_time_per_video', 
                'forum_activity', 'num_course_views', 'location_change']
    X = df[features]
    y = (df['quiz_accuracy'] < 50).astype(int)
    
    # Create model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    
    # Check cross-validation results
    assert cv_scores.mean() >= 0.75, f"Mean cross-validation accuracy {cv_scores.mean()} below threshold of 0.75"
    assert cv_scores.min() >= 0.65, f"Minimum cross-validation accuracy {cv_scores.min()} below threshold of 0.65"
    
    # Train full model for additional analysis
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Calculate detailed metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
    
    # Check F1 score
    assert f1 >= 0.75, f"F1 score {f1} below threshold of 0.75"

def test_anomaly_detection_quality(generate_synthetic_data):
    """Test quality of anomaly detection"""
    df = generate_synthetic_data
    
    # Apply anomaly detection
    df_result, anomaly_df = detect_anomalies(df)
    
    # Calculate ground truth based on our data generation (quiz_accuracy < 50)
    ground_truth_anomalies = (df['quiz_accuracy'] < 50).astype(int)
    
    # Compare with detected anomalies
    detected_anomalies = df_result['combined_anomaly']
    
    # Calculate accuracy between ground truth and detected anomalies
    accuracy = accuracy_score(ground_truth_anomalies, detected_anomalies)
    
    # Calculate detailed metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth_anomalies, 
        detected_anomalies, 
        average='binary'
    )
    
    # Print metrics for debugging
    print(f"Anomaly detection: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    # Assertions
    assert accuracy >= 0.7, f"Anomaly detection accuracy {accuracy} below threshold of 0.7"
    assert precision >= 0.6, f"Anomaly detection precision {precision} below threshold of 0.6"
    assert recall >= 0.6, f"Anomaly detection recall {recall} below threshold of 0.6"