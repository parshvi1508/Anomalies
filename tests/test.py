import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from model_utils2 import train_model, detect_anomalies, identify_at_risk_students

# Sample test data fixture
@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    data = {
        'student_id': range(1, 101),
        'video_completion_rate': np.random.uniform(0, 100, 100),
        'quiz_accuracy': np.random.uniform(0, 100, 100),
        'avg_time_per_video': np.random.uniform(5, 60, 100),
        'forum_activity': np.random.randint(0, 10, 100),
        'num_course_views': np.random.randint(10, 100, 100),
        'location_change': np.random.randint(0, 10, 100)
    }
    return pd.DataFrame(data)

@pytest.fixture
def realistic_data():
    """Create more realistic data with correlations for testing"""
    np.random.seed(42)
    n_samples = 200
    data = {
        'student_id': range(1, n_samples+1),
        'video_completion_rate': np.random.uniform(0, 100, n_samples),
        'quiz_accuracy': np.random.uniform(0, 100, n_samples),
        'avg_time_per_video': np.random.uniform(5, 60, n_samples),
        'forum_activity': np.random.randint(0, 10, n_samples),
        'num_course_views': np.random.randint(10, 100, n_samples),
        'location_change': np.random.randint(0, 10, n_samples)
    }
    
    # Create some correlations to make the data more realistic
    df = pd.DataFrame(data)
    
    # Make quiz accuracy correlate negatively with avg_time_per_video
    noise = np.random.normal(0, 10, n_samples)
    df['quiz_accuracy'] = 100 - df['avg_time_per_video'] + noise
    df['quiz_accuracy'] = df['quiz_accuracy'].clip(0, 100)
    
    # Make video_completion_rate positively correlate with quiz_accuracy
    noise = np.random.normal(0, 15, n_samples)
    df['video_completion_rate'] = df['quiz_accuracy'] * 0.8 + noise
    df['video_completion_rate'] = df['video_completion_rate'].clip(0, 100)
    
    return df

def test_train_model_accuracy(sample_data):
    """Test that the trained model achieves minimum accuracy"""
    model, metrics = train_model(sample_data)
    
    # Test accuracy is above minimum threshold
    assert metrics['accuracy'] >= 0.6, f"Model accuracy {metrics['accuracy']} is below threshold of 0.6"
    
    # Test that confusion matrix is present
    assert 'confusion_matrix' in metrics, "Confusion matrix missing from metrics"
    
    # Test that feature importance was calculated
    assert 'feature_importance' in metrics, "Feature importance missing from metrics"

def test_detect_anomalies(sample_data):
    """Test that anomaly detection adds required columns"""
    df_result, anomaly_df = detect_anomalies(sample_data)

    # Test that all expected columns are present
    expected_columns = ['anomaly_score', 'is_anomaly', 'rule_based_anomaly',
                        'combined_anomaly', 'dropout_risk']
    for col in expected_columns:
        assert col in df_result.columns, f"Column {col} missing from anomaly detection results"  

    # Test that some anomalies are detected (basic check)
    anomaly_pct = df_result['combined_anomaly'].mean() * 100
    assert 0 < anomaly_pct < 100, f"Anomaly percentage {anomaly_pct}% should be between 0% and 100%"
    
    # More meaningful test: verify correlation between quiz scores and anomaly detection
    low_performers = df_result[df_result['quiz_accuracy'] < 50]
    high_performers = df_result[df_result['quiz_accuracy'] >= 70]
    
    if len(low_performers) > 0 and len(high_performers) > 0:
        low_anomaly_rate = low_performers['combined_anomaly'].mean()
        high_anomaly_rate = high_performers['combined_anomaly'].mean()
        
        # Low performers should be more likely to be flagged as anomalies
        assert low_anomaly_rate >= high_anomaly_rate, \
            f"Anomaly detection isn't prioritizing low performers: {low_anomaly_rate:.2f} vs {high_anomaly_rate:.2f}"       
def test_at_risk_identification(sample_data):
    """Test at-risk student identification"""
    at_risk_df = identify_at_risk_students(sample_data, threshold=0.5)
    
    # Test that risk_score column is present
    assert 'risk_score' in at_risk_df.columns, "Risk score column missing from at-risk results"
    
    # Test that all risk scores are above the threshold
    assert all(at_risk_df['risk_score'] >= 0.5), "Some risk scores below threshold"

def test_model_cross_validation(realistic_data):
    """Test model with cross-validation for more robust evaluation"""
    # Instead of creating data inside the function, use the fixture
    df = realistic_data
    
    # Prepare the target variable (simulating anomaly classification)
    # Students with low quiz scores and high video time are considered anomalies
    df['anomaly_target'] = ((df['quiz_accuracy'] < 40) & 
                           (df['avg_time_per_video'] > 40)).astype(int)
    
    # Run model training
    features = ['video_completion_rate', 'quiz_accuracy', 'avg_time_per_video', 
                'forum_activity', 'num_course_views', 'location_change']
    X = df[features]
    y = df['anomaly_target']
    
    # Create and train a model (using RandomForest as an example)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    
    # Check cross-validation mean score
    assert cv_scores.mean() >= 0.7, f"Cross-validation mean score {cv_scores.mean():.4f} below threshold of 0.7"
    
    # Train full model for metrics calculation
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Calculate feature importance
    feature_importance = {}
    for i, feature in enumerate(features):
        feature_importance[feature] = model.feature_importances_[i]
    
    # Get classification report as dictionary
    report = classification_report(y, y_pred, output_dict=True)
    
    # Create metrics dictionary similar to what train_model would return
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'classification_report': report,
        'feature_importance': feature_importance
    }
    
    # Check feature importance values sum to approximately 1
    feature_importance_sum = sum(metrics['feature_importance'].values())
    assert 0.95 <= feature_importance_sum <= 1.05, f"Feature importance sum {feature_importance_sum} not approximately 1"