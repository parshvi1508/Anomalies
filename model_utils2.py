import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_model(df):
    """Train a model on the student data to predict at-risk students"""
    features = ['video_completion_rate', 'quiz_accuracy', 'avg_time_per_video', 
                'forum_activity', 'num_course_views', 'location_change']
    
    # Create X (features) - ensure all features exist
    available_features = [f for f in features if f in df.columns]
    X = df[available_features]
    
    # Create target y - using low quiz scores as an indicator of at-risk students
    y = (df['quiz_accuracy'] < 50).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': model.score(X_test, y_test),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'feature_importance': dict(zip(available_features, model.feature_importances_)),
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    # Save model for later use
    try:
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/student_risk_model.pkl')
    except Exception as e:
        print(f"Could not save model: {str(e)}")
    
    # Return the model and metrics
    return model, metrics

def detect_anomalies(df):
    """Detect anomalies in the student data using multiple methods"""
    # Make a copy of the dataframe to avoid warnings
    df = df.copy()
    
    # Define some simple rule-based anomaly types
    rule_based_anomalies = df[
        ((df['avg_time_per_video'] > 40) & (df['quiz_accuracy'] < 50)) |  # High video time + low scores
        (df['forum_activity'] < 2) |  # Low forum activity
        ((df['video_completion_rate'] > 80) & (df['quiz_accuracy'] < 40)) |  # Video binging with low retention
        (df['location_change'] > 5)  # Excessive location hopping
    ]
    
    # Use Isolation Forest for algorithmic anomaly detection
    features = ['video_completion_rate', 'quiz_accuracy', 'avg_time_per_video', 
                'forum_activity', 'num_course_views']
    
    # Ensure all features exist in the dataframe
    available_features = [f for f in features if f in df.columns]
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[available_features])
    
    # Apply Isolation Forest
    model = IsolationForest(contamination=0.1, random_state=42)
    df['anomaly_score'] = model.fit_predict(scaled_data)
    
    # Mark anomalies (Isolation Forest returns -1 for anomalies)
    df['is_anomaly'] = (df['anomaly_score'] == -1).astype(int)
    
    # Combine results - rule-based or algorithmic
    df['rule_based_anomaly'] = df['student_id'].isin(rule_based_anomalies['student_id']).astype(int)
    df['combined_anomaly'] = ((df['is_anomaly'] == 1) | (df['rule_based_anomaly'] == 1)).astype(int)
    
    # Calculate feature contributions to anomalies
    anomaly_df = df[df['combined_anomaly'] == 1].copy()
    
    # Calculate dropout risk
    # For simplicity, we're using a weighted formula based on key risk factors
    df['dropout_risk'] = (
        (100 - df['quiz_accuracy']) * 0.5 +  # Low quiz scores increase risk
        (100 - df['video_completion_rate']) * 0.3 +  # Low completion increases risk
        (df['avg_time_per_video'] > 30).astype(int) * 10 +  # Long video times increase risk
        (df['forum_activity'] < 3).astype(int) * 15  # Low forum activity increases risk
    ) / 100.0
    
    # Cap at 100%
    df['dropout_risk'] = df['dropout_risk'].clip(0, 1.0)
    
    # Enhance risk score for detected anomalies
    df.loc[df['combined_anomaly'] == 1, 'dropout_risk'] = df.loc[df['combined_anomaly'] == 1, 'dropout_risk'].apply(lambda x: min(0.95, x * 1.25))
    
    return df, anomaly_df

def identify_at_risk_students(df, threshold=0.7):
    """Identify students at high risk of dropping out"""
    # Calculate weighted risk score
    risk_score = (
        (df['quiz_accuracy'] < 50).astype(int) * 0.35 +  # Low quiz scores
        (df['video_completion_rate'] < 60).astype(int) * 0.25 +  # Low video completion
        (df['forum_activity'] < 3).astype(int) * 0.2 +  # Low forum engagement
        (df['avg_time_per_video'] > 35).astype(int) * 0.1 +  # High time on videos (struggling or distracted)
        (df['location_change'] > 4).astype(int) * 0.1  # Frequent location changes (unstable learning environment)
    )
    
    # Identify high-risk students
    at_risk = df[risk_score >= threshold].copy()
    at_risk['risk_score'] = risk_score[risk_score >= threshold]
    
    return at_risk.sort_values('risk_score', ascending=False)

def generate_personalized_recommendations(student_data):
    """Generate personalized recommendations based on student data"""
    recommendations = []
    
    # Academic performance recommendations
    if student_data['quiz_accuracy'] < 50:
        recommendations.append("Schedule a one-on-one academic counseling session")
        recommendations.append("Provide supplementary materials focused on weak areas")
        
        if student_data['quiz_accuracy'] < 30:
            recommendations.append("Consider remedial coursework to build foundational skills")
    
    # Engagement recommendations
    if student_data['video_completion_rate'] < 60:
        recommendations.append("Break content into smaller, more digestible segments")
        recommendations.append("Add interactive elements to maintain attention")
    
    if student_data['forum_activity'] < 3:
        recommendations.append("Assign partner/group activities to encourage interaction")
        recommendations.append("Create discussion prompts related to student interests")
    
    # Time management recommendations
    if student_data['avg_time_per_video'] > 35:
        recommendations.append("Provide note-taking templates to focus attention")
        recommendations.append("Suggest time management techniques like Pomodoro")
    
    # Learning environment recommendations
    if student_data['location_change'] > 4:
        recommendations.append("Suggest creating a dedicated study environment")
        recommendations.append("Provide offline access to course materials")
    
    return recommendations



