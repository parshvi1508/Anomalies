import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
from course_recommendation import run_course_recommendation
from model_utils2 import detect_anomalies, train_model
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="ISM- E learning Analytics and Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #3366cc;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #5a5a5a;
        margin-top: 1.5rem;
    }
    .feature-card {
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .anomaly-card {
        background-color: #f0f7ff;
        border-left: 5px solid #3366cc;
    }
    .recommendation-card {
        background-color: #f5f5f5;
        border-left: 5px solid #ff6b6b;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #888888;
        font-size: 0.8rem;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

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

def run_anomaly_detection(reset_callback):
    """Run the anomaly detection section of the app"""
    st.markdown("<h1 class='sub-header'>🔍 ISM- E learning Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.button("← Back to Home", on_click=reset_callback)
    
    st.markdown("""
        This dashboard helps identify and analyze student learning patterns.
        Upload student activity data to begin the analysis and receive targeted recommendations.
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
                st.sidebar.markdown("## 🔍 Filter Options")
                
                # Score Range Filter
                score_range = st.sidebar.slider(
                    "Quiz Score Range (%)", 
                    min_value=0, 
                    max_value=100, 
                    value=(0, 100),
                    step=5
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
                    options=list(anomaly_types.keys()),
                    default=[]
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
                    st.markdown("<h3>📌 Anomalies Detected</h3>", unsafe_allow_html=True)
                    anomaly_count = len(filtered_df[filtered_df['combined_anomaly'] == 1])
                    st.success(f"Found {anomaly_count} potential anomalies out of {len(filtered_df)} students")
                    
                    # Make student_id column clickable
                    if not filtered_df.empty:
                        st.markdown("Click on a student ID to view detailed recommendations")
                        
                        # Sort dataframe to show anomalies first
                        filtered_df = filtered_df.sort_values('combined_anomaly', ascending=False)
                        
                        # Display dataframe with clickable links
                        df_display = filtered_df[['student_id', 'quiz_accuracy', 'video_completion_rate', 
                                               'forum_activity', 'combined_anomaly']].copy()
                        df_display.columns = ['Student ID', 'Quiz Score (%)', 'Video Completion (%)', 
                                             'Forum Posts', 'Anomaly Detected']
                        
                        selected_student = None
                        
                        # Create clickable dataframe
                        student_ids = df_display['Student ID'].tolist()
                        selected_index = st.selectbox("Select a student to view recommendations:", 
                                                   options=range(len(student_ids)),
                                                   format_func=lambda x: f"Student {student_ids[x]}")
                        selected_student = student_ids[selected_index]
                        
                        st.dataframe(df_display, use_container_width=True)
                    else:
                        st.warning("No students match the current filters")
                
                with col2:
                    st.markdown("<h3>📊 Quick Stats</h3>", unsafe_allow_html=True)
                    
                    # Create metrics with better styling
                    metrics_data = [
                        {"label": "Avg Quiz Score", "value": f"{filtered_df['quiz_accuracy'].mean():.1f}%"},
                        {"label": "Completion Rate", "value": f"{filtered_df['video_completion_rate'].mean():.1f}%"},
                        {"label": "Active Students", "value": len(filtered_df[filtered_df['forum_activity'] > 0])},
                        {"label": "Anomaly Rate", "value": f"{filtered_df['combined_anomaly'].mean()*100:.1f}%"}
                    ]
                    
                    for metric in metrics_data:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h4>{metric['label']}</h4>
                            <h2>{metric['value']}</h2>
                        </div>
                        """, unsafe_allow_html=True)

                # Model Metrics Section
                st.markdown("<h3>🧠 Model Performance</h3>", unsafe_allow_html=True)
                
                if 'confusion_matrix' in metrics and 'feature_importance' in metrics:
                    metric_col1, metric_col2 = st.columns([1, 1])
                    
                    with metric_col1:
                        # Confusion matrix visualization
                        cm = metrics['confusion_matrix']
                        fig = px.imshow(
                            cm, 
                            text_auto=True,
                            labels=dict(x="Predicted", y="Actual"),
                            x=['Not At-Risk', 'At-Risk'],
                            y=['Not At-Risk', 'At-Risk'],
                            color_continuous_scale='Blues',
                            title='Confusion Matrix'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with metric_col2:
                        # Feature importance visualization
                        feat_imp = pd.DataFrame({
                            'Feature': list(metrics['feature_importance'].keys()),
                            'Importance': list(metrics['feature_importance'].values())
                        }).sort_values('Importance', ascending=False)
                        
                        fig = px.bar(
                            feat_imp, 
                            y='Feature', 
                            x='Importance', 
                            orientation='h',
                            title='Feature Importance',
                            color='Importance',
                            color_continuous_scale='Viridis'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # Analysis Tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Engagement", "Time Analysis", 
                    "Location Patterns", "Test Scores"
                ])
                
                with tab1:
                    st.subheader("Student Engagement Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Course views histogram
                        fig_views = px.histogram(
                            filtered_df, 
                            x='num_course_views', 
                            nbins=20, 
                            color='combined_anomaly',
                            color_discrete_map={0: "blue", 1: "red"},
                            labels={"combined_anomaly": "Anomaly Detected", "num_course_views": "Course Views"},
                            title="Course View Distribution"
                        )
                        st.plotly_chart(fig_views, use_container_width=True)
                    
                    with col2:
                        # Forum activity histogram
                        fig_forum = px.histogram(
                            filtered_df, 
                            x='forum_activity', 
                            nbins=15, 
                            color='combined_anomaly',
                            color_discrete_map={0: "green", 1: "red"},
                            labels={"combined_anomaly": "Anomaly Detected", "forum_activity": "Forum Posts"},
                            title="Forum Activity Distribution"
                        )
                        st.plotly_chart(fig_forum, use_container_width=True)
                    
                    # Scatter plot for engagement vs completion
                    fig_engagement = px.scatter(
                        filtered_df, 
                        x='video_completion_rate', 
                        y='forum_activity',
                        color='quiz_accuracy',
                        size='num_course_views',
                        hover_data=['student_id'],
                        labels={
                            "video_completion_rate": "Video Completion (%)",
                            "forum_activity": "Forum Activity",
                            "quiz_accuracy": "Quiz Score (%)"
                        },
                        title="Student Engagement Patterns",
                        color_continuous_scale="Viridis"
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
                            color='combined_anomaly',
                            color_discrete_map={0: "blue", 1: "red"},
                            labels={"avg_time_per_video": "Avg Minutes per Video", "combined_anomaly": "Anomaly"},
                            title="Time Spent per Video"
                        )
                        st.plotly_chart(fig_time, use_container_width=True)
                    
                    with col2:
                        # Time vs quiz scores
                        fig_time_vs_score = px.scatter(
                            filtered_df,
                            x='avg_time_per_video',
                            y='quiz_accuracy',
                            color='combined_anomaly',
                            color_discrete_map={0: "blue", 1: "red"},
                            labels={
                                "avg_time_per_video": "Avg Minutes per Video",
                                "quiz_accuracy": "Quiz Score (%)",
                                "combined_anomaly": "Anomaly"
                            },
                            title="Video Time vs Quiz Performance"
                        )
                        # Add reference lines
                        fig_time_vs_score.add_hline(y=50, line_dash="dash", line_color="gray")
                        fig_time_vs_score.add_vline(x=20, line_dash="dash", line_color="gray")
                        st.plotly_chart(fig_time_vs_score, use_container_width=True)

                with tab3:
                    st.subheader("Location Patterns")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Location changes pie chart
                        location_counts = filtered_df['location_change'].value_counts().reset_index()
                        location_counts.columns = ['Changes', 'Count']
                        
                        fig_location = px.pie(
                            location_counts, 
                            values='Count', 
                            names='Changes',
                            title="Device Location Changes",
                            hole=0.4
                        )
                        st.plotly_chart(fig_location, use_container_width=True)
                    
                    with col2:
                        # Location vs engagement
                        fig_loc_engagement = px.scatter(
                            filtered_df,
                            x='location_change',
                            y='video_completion_rate',
                            color='quiz_accuracy',
                            size='forum_activity',
                            hover_data=['student_id'],
                            labels={
                                "location_change": "Location Changes",
                                "video_completion_rate": "Video Completion (%)",
                                "quiz_accuracy": "Quiz Score (%)"
                            },
                            title="Location Changes vs. Engagement",
                            color_continuous_scale="Viridis"
                        )
                        st.plotly_chart(fig_loc_engagement, use_container_width=True)

                with tab4:
                    st.subheader("Test Scores Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Score distribution histogram
                        filtered_df['score_category'] = pd.cut(
                            filtered_df['quiz_accuracy'],
                            bins=[0, 40, 60, 80, 100],
                            labels=['At Risk', 'Below Average', 'Good', 'Excellent']
                        )
                        
                        fig_scores = px.histogram(
                            filtered_df,
                            x='quiz_accuracy',
                            color='score_category',
                            category_orders={"score_category": ['At Risk', 'Below Average', 'Good', 'Excellent']},
                            labels={"quiz_accuracy": "Quiz Score (%)", "score_category": "Performance Level"},
                            title="Score Distribution"
                        )
                        st.plotly_chart(fig_scores, use_container_width=True)
                    
                    with col2:
                        # Correlation heatmap
                        corr_features = ['quiz_accuracy', 'video_completion_rate', 'avg_time_per_video', 
                                       'forum_activity', 'num_course_views']
                        corr_matrix = filtered_df[corr_features].corr().round(2)
                        
                        fig_corr = px.imshow(
                            corr_matrix,
                            text_auto=True,
                            labels={"x": "Features", "y": "Features", "color": "Correlation"},
                            title="Feature Correlation Matrix",
                            color_continuous_scale="RdBu_r"
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)

                # Student Recommendations Section
                if 'selected_student' in locals() and selected_student is not None:
                    st.markdown("<h3>🎯 Personalized Student Recommendations</h3>", unsafe_allow_html=True)
                    
                    student_data = filtered_df[filtered_df['student_id'] == selected_student].iloc[0]
                    is_anomaly = student_data['combined_anomaly'] == 1
                    
                    # Profile Section
                    st.markdown(f"### Student {selected_student} Profile")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        status = "🔴 At Risk" if is_anomaly else "🟢 Normal"
                        st.markdown(f"**Status:** {status}")
                        st.markdown(f"**Quiz Score:** {student_data['quiz_accuracy']:.1f}%")
                        st.markdown(f"**Forum Posts:** {int(student_data['forum_activity'])}")
                    
                    with col2:
                        st.markdown(f"**Video Completion:** {student_data['video_completion_rate']:.1f}%")
                        st.markdown(f"**Avg Time per Video:** {student_data['avg_time_per_video']:.1f} min")
                        st.markdown(f"**Course Views:** {int(student_data['num_course_views'])}")
                    
                    with col3:
                        st.markdown(f"**Location Changes:** {int(student_data['location_change'])}")
                        dropout_risk = student_data.get('dropout_risk', None)
                        if dropout_risk is not None:
                            st.markdown(f"**Dropout Risk:** {dropout_risk:.1%}")
                        
                        # Determine risk level
                        if student_data['quiz_accuracy'] < 40:
                            risk_level = "High"
                        elif student_data['quiz_accuracy'] < 60:
                            risk_level = "Medium"
                        else:
                            risk_level = "Low"
                        st.markdown(f"**Risk Level:** {risk_level}")
                    
                    # Recommendations based on specific student data
                    st.markdown("### 📋 Targeted Recommendations")
                    
                    # Identify issues
                    issues = []
                    
                    if student_data['quiz_accuracy'] < 50:
                        issues.append("low_quiz_scores")
                    
                    if student_data['video_completion_rate'] < 60:
                        issues.append("low_video_completion")
                    
                    if student_data['forum_activity'] < 3:
                        issues.append("low_engagement")
                    
                    if student_data['avg_time_per_video'] > 40:
                        issues.append("high_video_time")
                    
                    if student_data['location_change'] > 5:
                        issues.append("high_location_changes")
                    
                    # Generate recommendations based on issues
                    def generate_student_recommendations(issues, student_data):
                        recommendations = {
                            "Learning Strategy": [],
                            "Engagement": [],
                            "Technical": []
                        }
                        
                        if "low_quiz_scores" in issues:
                            recommendations["Learning Strategy"].extend([
                                "Review course materials before attempting quizzes",
                                "Schedule regular study sessions",
                                "Consider joining study groups"
                            ])
                        
                        if "low_video_completion" in issues:
                            recommendations["Learning Strategy"].extend([
                                "Break video lectures into smaller segments",
                                "Take notes while watching videos",
                                "Set daily video watching goals"
                            ])
                        
                        if "low_engagement" in issues:
                            recommendations["Engagement"].extend([
                                "Participate in forum discussions regularly",
                                "Ask questions when concepts are unclear",
                                "Connect with other students"
                            ])
                        
                        if "high_video_time" in issues:
                            recommendations["Technical"].extend([
                                "Use video playback speed controls",
                                "Create a focused learning environment",
                                "Take structured breaks between videos"
                            ])
                        
                        if "high_location_changes" in issues:
                            recommendations["Technical"].extend([
                                "Establish a consistent study location",
                                "Ensure stable internet connection",
                                "Download materials for offline access"
                            ])
                        
                        return recommendations

                    recommendations = generate_student_recommendations(issues, student_data)
                    
                    # Display recommendations
                    for category, recs in recommendations.items():
                        with st.expander(f"{category} ({len(recs)} recommendations)", expanded=True):
                            for rec in recs:
                                st.markdown(f"• {rec}")
                    
                    # Quiz specific recommendations
                    if "low_quiz_scores" in issues:
                        with st.expander("📚 Recommended Course Content", expanded=True):
                            st.markdown("Based on quiz performance, we recommend the following resources:")
                            
                            # Course recommendations based on quiz score
                            def recommend_courses_based_on_quiz(quiz_score):
                                if quiz_score < 40:
                                    return [
            {"title": "Basic Concepts Review", "description": "A foundational course covering core concepts", "platform": "Byjus", "link": "#"},
            {"title": "Study Skills Fundamentals", "description": "Learn effective study techniques with personalized motivation", "platform": "Byjus", "link": "#"},
            {"title": "Interactive Practice Exercises", "description": "Additional practice problems with solutions and assessment", "platform": "Skillshare", "link": "#"}
        ]
                                elif quiz_score < 70:
                                    return [
            {"title": "Intermediate Practice", "description": "Targeted exercises with performance metrics", "platform": "Byjus", "link": "#"},
            {"title": "Topic Deep Dive", "description": "Detailed explanations aligned with learning objectives", "platform": "upGrad", "link": "#"},
            {"title": "Problem-Solving Strategies", "description": "Advanced techniques with engaging discussion forums", "platform": "Byjus", "link": "#"}
        ]
                                else:
                                    return [
            {"title": "Advanced Topics", "description": "Challenging material with comprehensive learning paths", "platform": "Udemy", "link": "#"},
            {"title": "Expert Level Content", "description": "Specialized concepts with adaptive learning styles", "platform": "edX", "link": "#"},
            {"title": "Mastery Projects", "description": "Apply knowledge with high engagement and satisfaction metrics", "platform": "upGrad", "link": "#"}
        ]
                            courses = recommend_courses_based_on_quiz(student_data['quiz_accuracy'])
                            
                            for i, course in enumerate(courses):
                                st.markdown(f"**{i+1}. {course['title']}**")
                                st.markdown(f"📋 {course['description']}")
                                st.markdown(f"🔗 [Access Course]({course['link']})")
                                st.markdown("---")
                    
                    # Student progress tracking
                    st.markdown("### 📈 Progress Tracking")
                    
                    # Create radar chart for student performance
                    categories = ['Quiz Scores', 'Video Completion', 'Forum Activity', 
                                'Video Time Efficiency', 'Location Stability']
                    
                    # Normalize values to 0-1 scale
                    values = [
                        student_data['quiz_accuracy'] / 100,
                        student_data['video_completion_rate'] / 100,
                        min(1.0, student_data['forum_activity'] / 10),  # Cap at 1.0
                        1.0 - min(1.0, (student_data['avg_time_per_video'] / 60)),  # Lower is better for time
                        1.0 - min(1.0, (student_data['location_change'] / 10))  # Lower is better for location changes
                    ]
                    
                    # Create radar chart
                    fig = go.Figure()
                    
                    # Add student data
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=f'Student {selected_student}'
                    ))
                    
                    # Add class average if we have the data
                    avg_values = [
                        filtered_df['quiz_accuracy'].mean() / 100,
                        filtered_df['video_completion_rate'].mean() / 100,
                        min(1.0, filtered_df['forum_activity'].mean() / 10),
                        1.0 - min(1.0, (filtered_df['avg_time_per_video'].mean() / 60)),
                        1.0 - min(1.0, (filtered_df['location_change'].mean() / 10))
                    ]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=avg_values,
                        theta=categories,
                        fill='toself',
                        name='Class Average'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )
                        ),
                        title="Student Performance Profile vs. Class Average",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interactive goal setting
                    st.markdown("### 🎯 Goal Setting")
                    
                    st.write("Set personalized improvement goals for this student:")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interactive goal setting
                    st.markdown("### 🎯 Goal Setting")
                    
                    st.write("Set personalized improvement goals for this student:")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        quiz_goal = st.slider(
                            "Quiz Score Goal (%)", 
                            min_value=int(student_data['quiz_accuracy']), 
                            max_value=100, 
                            value=min(int(student_data['quiz_accuracy'] + 15), 100)
                        )
                        
                        completion_goal = st.slider(
                            "Video Completion Goal (%)", 
                            min_value=int(student_data['video_completion_rate']), 
                            max_value=100, 
                            value=min(int(student_data['video_completion_rate'] + 20), 100)
                        )
                    
                    with col2:
                        forum_goal = st.slider(
                            "Forum Activity Goal (posts)", 
                            min_value=int(student_data['forum_activity']), 
                            max_value=20, 
                            value=min(int(student_data['forum_activity'] + 3), 20)
                        )
                        
                        if "low_video_completion" in issues:
                            time_goal = st.slider(
                                "Target Minutes per Video", 
                                min_value=10, 
                                max_value=int(student_data['avg_time_per_video']), 
                                value=max(20, int(student_data['avg_time_per_video']) - 10)
                            )
                    
                    # Display goal summary
                    st.markdown("#### Goal Summary")
                    
                    # Calculate improvement percentages
                    quiz_improvement = ((quiz_goal - student_data['quiz_accuracy']) / student_data['quiz_accuracy']) * 100
                    completion_improvement = ((completion_goal - student_data['video_completion_rate']) / max(1, student_data['video_completion_rate'])) * 100
                    forum_improvement = ((forum_goal - student_data['forum_activity']) / max(1, student_data['forum_activity'])) * 100
                    
                    # Create a progress tracking chart
                    improvement_data = {
                        'Metric': ['Quiz Score', 'Video Completion', 'Forum Activity'],
                        'Current': [student_data['quiz_accuracy'], student_data['video_completion_rate'], student_data['forum_activity']],
                        'Goal': [quiz_goal, completion_goal, forum_goal]
                    }
                    
                    df_goals = pd.DataFrame(improvement_data)
                    
                    fig_goals = go.Figure()
                    
                    for i, row in df_goals.iterrows():
                        fig_goals.add_trace(go.Bar(
                            name='Current',
                            x=[row['Metric']],
                            y=[row['Current']],
                            marker_color='royalblue'
                        ))
                        fig_goals.add_trace(go.Bar(
                            name='Goal',
                            x=[row['Metric']],
                            y=[row['Goal']],
                            marker_color='lightgreen'
                        ))
                    
                    # Customize layout
                    fig_goals.update_layout(
                        barmode='group',
                        title="Current Values vs. Goals",
                        yaxis_title="Value",
                        legend_title="Status"
                    )
                    
                    st.plotly_chart(fig_goals, use_container_width=True)
                    courses = recommend_courses_based_on_quiz(student_data['quiz_accuracy'])

                    for i, course in enumerate(courses):
                        st.markdown(f"**{i+1}. {course['title']}**")
                        st.markdown(f"📋 {course['description']}")
                        st.markdown(f"🏫 **Platform:** {course['platform']}")
                        st.markdown(f"🔗 [Access Course]({course['link']})")
                        st.markdown("---")
                    # Goal commitment
                    with st.form("goal_commitment"):
                        st.markdown("#### Commit to these goals")
                        notes = st.text_area("Add notes or action items:", 
                                            placeholder="e.g., Schedule weekly check-ins, provide additional resources...")
                        deadline = st.date_input("Target completion date:")
                        
                        submitted = st.form_submit_button("Save Goals")
                        if submitted:
                            st.success(f"✅ Goals for Student {selected_student} have been saved!")
                            
                            # Here you would typically save the goals to a database
                            st.balloons()
