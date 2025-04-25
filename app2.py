import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from model_utils import detect_anomalies, train_model
import os
import plotly.express as px 
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="E-Learning Anomaly Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and Description
st.title("📊 E-Learning Anomaly Detection Dashboard")
st.markdown("""
    This dashboard helps identify and analyze student learning patterns.
    Upload your student data to begin the analysis.
""")

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

# File Upload
uploaded_file = st.file_uploader("📁 Upload Student Data CSV", type="csv")

if uploaded_file:
    with st.spinner('Processing data...'):
        # Load and validate data
        df = load_and_process_data(uploaded_file)
        
        if df is not None:
            # Train model and detect anomalies
            st.info('🔄 Training model...')
            trained_df = train_model(df)
            
            st.info('🔍 Detecting anomalies...')
            anomalies = detect_anomalies(df)
            
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

            # Analysis Tabs with filtered data
            tab1, tab2, tab3, tab4 = st.tabs([
                "Engagement", "Time Analysis", 
                "Location Patterns", "Test Scores"
            ])
            
            with tab1:
                fig_engagement = px.histogram(
                    filtered_df, 
                    x='num_course_views', 
                    nbins=20, 
                    title="Course View Distribution"
                )
                st.plotly_chart(fig_engagement, use_container_width=True, key="engagement_hist")

            with tab2:
                fig_time = px.box(
                    filtered_df, 
                    y='avg_time_per_video', 
                    points="all", 
                    hover_data=['student_id'],
                    title="Time Spent per Video"
                )
                st.plotly_chart(fig_time, use_container_width=True, key="time_box")

            with tab3:
                fig_location = px.pie(
                    filtered_df, 
                    names='location_change', 
                    title="Device Switching Pattern"
                )
                st.plotly_chart(fig_location, use_container_width=True, key="location_pie")

            with tab4:
                # Score categories
                filtered_df['score_category'] = pd.cut(
                    filtered_df['quiz_accuracy'],
                    bins=[0, 40, 60, 80, 100],
                    labels=['At Risk', 'Below Average', 'Good', 'Excellent']
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_scores = px.histogram(
                        filtered_df,
                        x='quiz_accuracy',
                        color='score_category',
                        title="Score Distribution",
                        labels={'quiz_accuracy': 'Quiz Score (%)'}
                    )
                    st.plotly_chart(fig_scores, use_container_width=True, key="score_hist")
                
                with col2:
                    correlation_matrix = filtered_df[[
                        'quiz_accuracy', 'video_completion_rate',
                        'avg_time_per_video', 'forum_activity'
                    ]].corr()
                    fig_corr = px.imshow(
                        correlation_matrix,
                        title="Correlation Analysis"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True, key="corr_matrix")

            # Risk Analysis
            st.subheader("⚠️ Risk Analysis")
            
            # Prepare features for risk prediction
            risk_features = ['quiz_accuracy', 'video_completion_rate',
                           'avg_time_per_video', 'forum_activity', 'num_course_views']
            X = filtered_df[risk_features]
            y = (filtered_df['quiz_accuracy'] < 40) & (filtered_df['video_completion_rate'] < 50)
            
            # Train model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Predict dropout risk
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
                
                # Recommendations
                st.subheader("📋 Automated Recommendations")
                for _, student in high_risk.iterrows():
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
                        # Add personalized learning path recommendations
                        st.info("\n".join(recommendations))
                        
                        # Personalized Learning Path section
                        st.subheader("🛤️ Personalized Learning Path")
                        
                        if student['quiz_accuracy'] >= 70:
                            st.success("**Advanced Learner**")
                            learning_path = [
                                "• Offer accelerated tracks with optional advanced modules",
                                "• Unlock challenge tasks and badges",
                                "• Award badges and certificates for milestone achievements",
                                "• Provide access to additional research materials",
                                "• Connect with industry mentors"
                            ]
                            
                        elif student['quiz_accuracy'] >= 40:
                            st.warning("**Intermediate Learner**")
                            learning_path = [
                                "• Provide guided paths with a mix of practice and theory",
                                "• Introduce checkpoints for concept reinforcement",
                                "• Award badges and certificates for milestone achievements",
                                "• Offer personalized feedback highlighting strengths and improvement areas",
                                "• Schedule regular progress check-ins"
                            ]
                            
                        else:
                            st.error("**Needs Additional Support**")
                            learning_path = [
                                "• Break content into smaller, digestible chunks",
                                "• Provide additional learning support and repetition where needed",
                                "• Assign mentors or study buddies for personal guidance",
                                "• Award badges and certificates for milestone achievements",
                                "• Offer personalized feedback highlighting strengths and improvement areas",
                                "• Schedule weekly check-ins with instructors"
                            ]
                        
                        st.markdown("\n".join(learning_path))
                        
                        
                        # Risk factors
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Quiz Score", f"{student['quiz_accuracy']:.1f}%")
                            st.metric("Video Completion", f"{student['video_completion_rate']:.1f}%")
                        with col2:
                            st.metric("Forum Activity", f"{student['forum_activity']} posts")
                            st.metric("Course Views", student['num_course_views'])

            # Save model
            try:
                model_dir = os.path.join(os.path.dirname(__file__), 'models')
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, 'dropout_predictor.pkl')
                joblib.dump(model, model_path)
            except Exception as e:
                st.warning(f"⚠️ Could not save model: {str(e)}")    