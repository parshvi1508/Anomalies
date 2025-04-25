import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import plotly.express as px 
import plotly.graph_objects as go
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from xourse_recommendation import run_course_recommendation  # Fixed import
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="ISM- E learning Analytics and Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App state management
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def reset_to_home():
    st.session_state.page = 'home'

def navigate_to_anomalies():
    st.session_state.page = 'anomalies'

def navigate_to_recommendations():
    st.session_state.page = 'recommendations'

def load_and_process_data(uploaded_file):
    """Load and validate the uploaded data"""
    try:
        df = pd.read_csv(uploaded_file, encoding='latin-1')
        required_columns = [
            'student_id', 'video_completion_rate', 'quiz_accuracy',
            'avg_time_per_video', 'forum_activity', 'location_change',
            'num_course_views'
        ]
        if not all(col in df.columns for col in required_columns):
            st.error("❌ Missing required columns in the CSV file")
            return None
        return df
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None

def train_model(df):
    """Train a model on the student data"""
    features = ['video_completion_rate', 'quiz_accuracy', 'avg_time_per_video', 
                'forum_activity', 'num_course_views']
    
    # Create X (features)
    X = df[features]
    
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
        'feature_importance': dict(zip(features, model.feature_importances_)),
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    # Save model for later use
    try:
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/student_risk_model.pkl')
    except Exception as e:
        st.warning(f"Could not save model: {str(e)}")
    
    # Return the model and metrics
    return model, metrics

def detect_anomalies(df):
    """Detect anomalies in the student data using multiple methods"""
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
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    
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
    
    return df, anomaly_df

def run_anomaly_detection(reset_callback):
    """Run the anomaly detection section of the app"""
    st.title("📊 ISM- E learning Analytics Dashboard")
    st.button("← Back to Home", on_click=reset_callback)
    
    st.markdown("""
        This dashboard helps identify and analyze student learning patterns.
        Upload your student data to begin the analysis.
    """)

    # File Upload
    uploaded_file = st.file_uploader("📁 Upload Student Data CSV", type="csv")

    if uploaded_file:
        with st.spinner('Processing data...'):
            # Load and validate data
            df = load_and_process_data(uploaded_file)
            
            if df is not None:
                # Train model and detect anomalies
                st.info('🔄 Training predictive model...')
                model, metrics = train_model(df)
                
                st.info('🔍 Detecting anomalies...')
                df, anomalies = detect_anomalies(df)
                
                # Sidebar Filters
                st.sidebar.subheader("🔍 Filter Options")
                
                # Score Range Filter
                score_range = st.sidebar.slider(
                    "Quiz Score Range (%)", 
                    min_value=0, 
                    max_value=100, 
                    value=(0, 100)
                )

                # Engagement Level Filter
                engagement_levels = st.sidebar.multiselect(
                    "Engagement Level",
                    options=['Low', 'Medium', 'High'],
                    default=['Low', 'Medium', 'High']
                )

                # Forum Activity Filter
                forum_activity = st.sidebar.slider(
                    "Minimum Forum Posts",
                    min_value=0,
                    max_value=int(df['forum_activity'].max()),
                    value=0
                )

                # Anomaly Types Filter
                anomaly_types = {
                    'High Video Time + Low Scores': (df['avg_time_per_video'] > 40) & (df['quiz_accuracy'] < 50),
                    'Low Forum Activity': df['forum_activity'] < 2,
                    'Video Binging': (df['video_completion_rate'] > 80) & (df['quiz_accuracy'] < 40),
                    'Location Hopper': df['location_change'] > 5
                }
                
                selected_anomalies = st.sidebar.multiselect(
                    '⚠️ Select Anomaly Types:',
                    options=list(anomaly_types.keys())
                )

                # Apply Filters
                filtered_df = df.copy()

                # Apply score range filter
                filtered_df = filtered_df[
                    (filtered_df['quiz_accuracy'] >= score_range[0]) & 
                    (filtered_df['quiz_accuracy'] <= score_range[1])
                ]

                # Apply engagement level filter
                engagement_conditions = {
                    'Low': (df['video_completion_rate'] < 30),
                    'Medium': (df['video_completion_rate'] >= 30) & (df['video_completion_rate'] < 70),
                    'High': (df['video_completion_rate'] >= 70)
                }
                
                if engagement_levels:
                    engagement_mask = pd.Series(False, index=df.index)
                    for level in engagement_levels:
                        engagement_mask |= engagement_conditions[level]
                    filtered_df = filtered_df[engagement_mask]

                # Apply forum activity filter
                filtered_df = filtered_df[filtered_df['forum_activity'] >= forum_activity]

                # Apply anomaly type filters
                if selected_anomalies:
                    anomaly_mask = pd.Series(False, index=df.index)
                    for anomaly_type in selected_anomalies:
                        anomaly_mask |= anomaly_types[anomaly_type]
                    filtered_df = filtered_df[anomaly_mask]

                # Display results with filtered data
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📌 Anomalies Detected")
                    st.success(f"Found {len(anomalies)} potential anomalies")
                    st.dataframe(filtered_df, use_container_width=True)
                
                with col2:
                    st.subheader("📊 Quick Stats")
                    st.metric("Avg Quiz Score", f"{filtered_df['quiz_accuracy'].mean():.1f}%")
                    st.metric("Completion Rate", f"{filtered_df['video_completion_rate'].mean():.1f}%")
                    st.metric("Active Students", len(filtered_df[filtered_df['forum_activity'] > 0]))
                    st.metric("Anomaly Rate", f"{filtered_df['combined_anomaly'].mean()*100:.1f}%")

                # Model Metrics Section
                st.subheader("🧠 Predictive Model Performance")
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("Model Accuracy", f"{metrics['accuracy']:.2f}")
                    st.write("Feature Importance:")
                    # Create horizontal bar chart for feature importance
                    feat_imp = pd.DataFrame({
                        'Feature': list(metrics['feature_importance'].keys()),
                        'Importance': list(metrics['feature_importance'].values())
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        feat_imp, 
                        y='Feature', 
                        x='Importance', 
                        orientation='h',
                        title='Feature Importance'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with metric_col2:
                    # Precision, Recall, F1 Score
                    cr = metrics['classification_report']
                    st.write("Classification Report:")
                    
                    # Create a dataframe from the classification report for class 1 (at-risk students)
                    cr_df = pd.DataFrame({
                        'Metric': ['Precision', 'Recall', 'F1-Score'],
                        'Value': [cr['1']['precision'], cr['1']['recall'], cr['1']['f1-score']]
                    })
                    
                    fig = px.bar(
                        cr_df, 
                        x='Metric', 
                        y='Value', 
                        title='Model Metrics for At-Risk Students',
                        color='Value',
                        color_continuous_scale='RdYlGn_r'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with metric_col3:
                    # Confusion Matrix
                    st.write("Confusion Matrix:")
                    cm = metrics['confusion_matrix']
                    
                    # Create confusion matrix heatmap
                    fig = px.imshow(
                        cm, 
                        text_auto=True,
                        labels=dict(x="Predicted", y="Actual"),
                        x=['Not At-Risk', 'At-Risk'],
                        y=['Not At-Risk', 'At-Risk'],
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(title='Confusion Matrix')
                    st.plotly_chart(fig, use_container_width=True)

                # Analysis Tabs with filtered data
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Engagement", "Time Analysis", 
                    "Location Patterns", "Test Scores"
                ])
                
                with tab1:
                    st.subheader("Student Engagement Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Course views distribution
                        fig_views = px.histogram(
                            filtered_df, 
                            x='num_course_views', 
                            nbins=20, 
                            title="Course View Distribution",
                            color='combined_anomaly',
                            color_discrete_map={0: "blue", 1: "red"},
                            labels={"combined_anomaly": "Anomaly Detected"}
                        )
                        st.plotly_chart(fig_views, use_container_width=True)
                    
                    with col2:
                        # Forum activity
                        fig_forum = px.histogram(
                            filtered_df, 
                            x='forum_activity', 
                            nbins=15, 
                            title="Forum Activity Distribution",
                            color='combined_anomaly',
                            color_discrete_map={0: "green", 1: "red"},
                            labels={"combined_anomaly": "Anomaly Detected"}
                        )
                        st.plotly_chart(fig_forum, use_container_width=True)
                    
                    # Overall engagement scatter plot
                    fig_engagement = px.scatter(
                        filtered_df, 
                        x='video_completion_rate', 
                        y='forum_activity',
                        color='combined_anomaly',
                        color_discrete_map={0: "blue", 1: "red"},
                        size='num_course_views',
                        hover_data=['student_id', 'quiz_accuracy'],
                        title="Student Engagement Pattern",
                        labels={"combined_anomaly": "Anomaly Detected", 
                               "video_completion_rate": "Video Completion (%)",
                               "forum_activity": "Forum Posts"}
                    )
                    st.plotly_chart(fig_engagement, use_container_width=True)

                with tab2:
                    st.subheader("Time Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Box plot of time per video
                        fig_time = px.box(
                            filtered_df, 
                            y='avg_time_per_video', 
                            points="all", 
                            hover_data=['student_id'],
                            title="Time Spent per Video",
                            color='combined_anomaly',
                            color_discrete_map={0: "green", 1: "red"},
                            labels={"combined_anomaly": "Anomaly Detected"}
                        )
                        st.plotly_chart(fig_time, use_container_width=True)
                    
                    with col2:
                        # Scatter plot: time vs quiz scores
                        fig_time_vs_score = px.scatter(
                            filtered_df,
                            x='avg_time_per_video',
                            y='quiz_accuracy',
                            color='combined_anomaly',
                            color_discrete_map={0: "blue", 1: "red"},
                            title="Video Time vs Quiz Scores",
                            labels={"combined_anomaly": "Anomaly Detected",
                                   "avg_time_per_video": "Avg Time per Video (min)",
                                   "quiz_accuracy": "Quiz Score (%)"}
                        )
                        # Add quadrant lines
                        fig_time_vs_score.add_hline(y=50, line_dash="dash", line_color="gray")
                        fig_time_vs_score.add_vline(x=20, line_dash="dash", line_color="gray")
                        st.plotly_chart(fig_time_vs_score, use_container_width=True)
                    
                    # Time density by anomaly status
                    fig_time_density = px.violin(
                        filtered_df,
                        x='combined_anomaly',
                        y='avg_time_per_video',
                        box=True,
                        points="all",
                        color='combined_anomaly',
                        color_discrete_map={0: "blue", 1: "red"},
                        labels={"combined_anomaly": "Anomaly Detected",
                               "avg_time_per_video": "Avg Time per Video (min)"},
                        title="Time Distribution by Anomaly Status"
                    )
                    st.plotly_chart(fig_time_density, use_container_width=True)

                with tab3:
                    st.subheader("Location and Device Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Location changes pie chart
                        fig_location = px.pie(
                            filtered_df, 
                            names='location_change', 
                            title="Device Switching Pattern",
                            hole=0.4
                        )
                        st.plotly_chart(fig_location, use_container_width=True)
                    
                    with col2:
                        # Location change vs engagement
                        fig_loc_vs_engagement = px.scatter(
                            filtered_df,
                            x='location_change',
                            y='video_completion_rate',
                            color='combined_anomaly',
                            color_discrete_map={0: "blue", 1: "red"},
                            size='num_course_views',
                            hover_data=['student_id', 'quiz_accuracy'],
                            title="Location Changes vs. Video Completion",
                            labels={"combined_anomaly": "Anomaly Detected"}
                        )
                        st.plotly_chart(fig_loc_vs_engagement, use_container_width=True)
                    
                    # Heatmap of correlations
                    correlation_features = ['video_completion_rate', 'quiz_accuracy', 
                                          'avg_time_per_video', 'forum_activity', 
                                          'location_change', 'num_course_views']
                    corr_matrix = filtered_df[correlation_features].corr().round(2)
                    
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        title="Feature Correlation Matrix",
                        color_continuous_scale="RdBu_r"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)

                with tab4:
                    st.subheader("Quiz Performance Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Score categories
                        filtered_df['score_category'] = pd.cut(
                            filtered_df['quiz_accuracy'],
                            bins=[0, 40, 60, 80, 100],
                            labels=['At Risk', 'Below Average', 'Good', 'Excellent']
                        )
                        
                        fig_scores = px.histogram(
                            filtered_df,
                            x='quiz_accuracy',
                            color='score_category',
                            title="Score Distribution",
                            labels={'quiz_accuracy': 'Quiz Score (%)'},
                            category_orders={"score_category": ['At Risk', 'Below Average', 'Good', 'Excellent']}
                        )
                        st.plotly_chart(fig_scores, use_container_width=True)
                    
                    with col2:
                        # Completion rate vs quiz score
                        fig_completion_vs_score = px.scatter(
                            filtered_df,
                            x='video_completion_rate',
                            y='quiz_accuracy',
                            color='score_category',
                            title="Video Completion vs. Quiz Scores",
                            labels={'video_completion_rate': 'Video Completion (%)', 
                                   'quiz_accuracy': 'Quiz Score (%)'},
                            category_orders={"score_category": ['At Risk', 'Below Average', 'Good', 'Excellent']}
                        )
                        # Add trendline
                        fig_completion_vs_score.update_layout(showlegend=True)
                        st.plotly_chart(fig_completion_vs_score, use_container_width=True)
                    
                    # ROC Curve and Precision-Recall Curve
                    st.subheader("Model Performance Curves")
                    
                    # Calculate and plot Precision-Recall curve
                    from sklearn.metrics import precision_recall_curve, roc_curve, auc
                    
                    # Create two columns for the charts
                    curve_col1, curve_col2 = st.columns(2)
                    
                    with curve_col1:
                        # Precision-Recall Curve
                        precision, recall, thresholds = precision_recall_curve(
                            metrics['y_test'], metrics['y_prob']
                        )
                        
                        fig_prc = go.Figure()
                        fig_prc.add_trace(go.Scatter(
                            x=recall, y=precision,
                            mode='lines',
                            name='Precision-Recall Curve'
                        ))
                        fig_prc.update_layout(
                            title='Precision-Recall Curve',
                            xaxis_title='Recall',
                            yaxis_title='Precision',
                            yaxis=dict(range=[0, 1.05]),
                            xaxis=dict(range=[0, 1.05])
                        )
                        st.plotly_chart(fig_prc, use_container_width=True)
                    
                    with curve_col2:
                        # ROC Curve
                        fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_prob'])
                        roc_auc = auc(fpr, tpr)
                        
                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(
                            x=fpr, y=tpr,
                            mode='lines',
                            name=f'ROC Curve (AUC = {roc_auc:.3f})'
                        ))
                        fig_roc.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            mode='lines',
                            line=dict(dash='dash', color='gray'),
                            name='Random Classifier'
                        ))
                        fig_roc.update_layout(
                            title='ROC Curve',
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate',
                            yaxis=dict(range=[0, 1.05]),
                            xaxis=dict(range=[0, 1.05])
                        )
                        st.plotly_chart(fig_roc, use_container_width=True)

                # Risk Analysis
                st.subheader("⚠️ Risk Analysis")
                
                # Prepare features for risk prediction
                risk_features = ['quiz_accuracy', 'video_completion_rate',
                               'avg_time_per_video', 'forum_activity', 'num_course_views']
                X = filtered_df[risk_features]
                
                # Use our model to predict dropout risk
                try:
                    filtered_df['dropout_risk'] = model.predict_proba(X)[:, 1]
                    high_risk = filtered_df[filtered_df['dropout_risk'] > 0.7].sort_values('dropout_risk', ascending=False)
                    
                    st.warning(f"Found {len(high_risk)} students at high risk of dropping out")
                    
                    if not high_risk.empty:
                        st.dataframe(
                            high_risk[[
                                'student_id', 'quiz_accuracy',
                                'video_completion_rate', 'dropout_risk'
                            ]],
                            use_container_width=True
                        )
                        
                        # Risk factors visualization
                        fig_risk = px.scatter(
                            high_risk,
                            x='video_completion_rate',
                            y='quiz_accuracy',
                            size='dropout_risk',
                            color='dropout_risk',
                            hover_data=['student_id', 'forum_activity'],
                            title='High Risk Students Analysis',
                            labels={
                                'video_completion_rate': 'Video Completion (%)',
                                'quiz_accuracy': 'Quiz Score (%)',
                                'dropout_risk': 'Dropout Risk'
                            },
                            color_continuous_scale='Reds'
                        )
                        st.plotly_chart(fig_risk, use_container_width=True)
                        
                        # Recommendations for high-risk students
                        st.subheader("📋 Automated Recommendations")
                        for _, student in high_risk.head(5).iterrows():
                            with st.expander(f"Student {student['student_id']} - Risk: {student['dropout_risk']:.1%}"):
                                recommendations = []
                                
                                if student['quiz_accuracy'] < 40:
                                    recommendations.extend([
                                        "• Schedule immediate academic counseling",
                                        "• Enroll in study skills workshop",
                                        "• Set up weekly progress checks"
                                    ])
                                
                                if student['video_completion_rate'] < 50:
                                    recommendations.extend([
                                        "• Review course material accessibility",
                                        "• Check for technical issues",
                                        "• Consider alternative content formats"
                                    ])
                                
                                if student['forum_activity'] < 2:
                                    recommendations.extend([
                                        "• Encourage peer interaction",
                                        "• Assign study group",
                                        "• Schedule collaborative sessions"
                                    ])
                                
                                st.info("\n".join(recommendations))
                                
                                # Individual student radar chart
                                categories = ['Quiz Score', 'Video Completion', 'Forum Activity',
                                           'Course Engagement', 'Location Stability']
                                values = [
                                    student['quiz_accuracy'] / 100,
                                    student['video_completion_rate'] / 100,
                                    min(1.0, student['forum_activity'] / 10),
                                    min(1.0, student['num_course_views'] / 50),
                                    1.0 - (min(1.0, student['location_change'] / 10))
                                ]
                                
                                fig_radar = go.Figure()
                                fig_radar.add_trace(go.Scatterpolar(
                                    r=values,
                                    theta=categories,
                                    fill='toself',
                                    name='Student Profile'
                                ))
                                fig_radar.update_layout(
                                    polar=dict(
                                        radialaxis=dict(
                                            visible=True,
                                            range=[0, 1]
                                        )
                                    ),
                                    showlegend=False,
                                    title='Student Engagement Profile'
                                )
                                st.plotly_chart(fig_radar, use_container_width=True)
                except Exception as e:
                    st.error(f"Error predicting risk: {str(e)}")

# Home Page
if st.session_state.page == 'home':
    st.title("📚 ISM- E learning Analytics and Recommendation System")
    
    st.markdown("""
    ### Welcome to the ISM- E learning Analytics and Recommendation System
    
    This platform offers two main features:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("### 🔍 Anomaly Detection")
        st.markdown("""
        Identify students who may be struggling or showing unusual learning patterns.
        Get personalized recommendations to support them.
        """)
        st.button("Anomaly Detection Dashboard", on_click=navigate_to_anomalies, use_container_width=True)
    
    with col2:
        st.success("### 📚 Course Recommendations")
        st.markdown("""
        Find the perfect courses for students based on their learning patterns, 
        interests, and academic needs.
        """)
        st.button("Course Recommendation Engine", on_click=navigate_to_recommendations, use_container_width=True)
    
    st.image("https://img.freepik.com/free-vector/online-tutorials-concept_23-2148688910.jpg", 
             caption="E-Learning Analytics")

# Anomaly Detection Page
elif st.session_state.page == 'anomalies':
    run_anomaly_detection(reset_to_home)

# Course Recommendation Page
elif st.session_state.page == 'recommendations':
    run_course_recommendation(reset_to_home)

if __name__ == "__main__":
    # This will run when the script is executed directly
    pass