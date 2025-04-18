import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

MODEL_PATH = "anomaly_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURE_COLS = [
    'avg_time_per_video',
    'num_course_views',
    'num_recommend_clicks',
    'quiz_accuracy',
    'forum_activity',
    'location_change',
    'total_videos_available',    
    'videos_watched',           
    'video_completion_rate'      
]

def generate_sample_data():
    np.random.seed(42)
    n_samples = 100
    
    # Generate student IDs
    student_ids = [f"STU{i:03d}" for i in range(1, n_samples + 1)]
    
    # Total videos in course (fixed at 50)
    total_videos = np.full(n_samples, 50)
    
    # Videos watched (with some realistic patterns)
    videos_watched = np.zeros(n_samples)
    # Regular students (60%)
    videos_watched[:60] = np.random.randint(35, 51, 60)
    # Struggling students (30%)
    videos_watched[60:90] = np.random.randint(10, 35, 30)
    # Inactive students (10%)
    videos_watched[90:] = np.random.randint(0, 10, 10)
    
    # Calculate completion rate
    completion_rate = (videos_watched / total_videos) * 100
    
    # Generate correlated quiz scores
    quiz_accuracy = np.zeros(n_samples)
    for i in range(n_samples):
        base_score = completion_rate[i] * 0.7  # Base score correlates with video completion
        variation = np.random.normal(0, 15)    # Add some random variation
        quiz_accuracy[i] = max(0, min(100, base_score + variation))  # Clip between 0-100
    
    return pd.DataFrame({
        'student_id': student_ids,
        'avg_time_per_video': np.where(
            videos_watched > 0,
            np.random.uniform(5, 45, n_samples),  # Active students
            np.random.uniform(0, 5, n_samples)    # Inactive students
        ),
        'num_course_views': np.where(
            videos_watched > 30,
            np.random.randint(40, 100, n_samples),  # Active students
            np.random.randint(5, 40, n_samples)     # Less active students
        ),
        'num_recommend_clicks': np.random.randint(0, 30, n_samples),
        'quiz_accuracy': quiz_accuracy,
        'forum_activity': np.where(
            completion_rate > 50,
            np.random.randint(10, 50, n_samples),  # Active students
            np.random.randint(0, 10, n_samples)    # Less active students
        ),
        'location_change': np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2]),
        'total_videos_available': total_videos,
        'videos_watched': videos_watched,
        'video_completion_rate': completion_rate
    })

def train_model(df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLS])
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    df['anomaly_score'] = model.fit_predict(X_scaled)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return df

def detect_anomalies(df):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(df[FEATURE_COLS])
    df['anomaly_score'] = model.predict(X_scaled)
    return df[df['anomaly_score'] == -1]