import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
from recommend import run_course_recommendation

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


# Function to generate personalized recommendations
def generate_student_recommendations(issues, student_data):
    """Generate personalized recommendations based on identified issues"""
    recommendations = {}
    
    if "low_quiz_scores" in issues:
        quiz_score = student_data['quiz_accuracy']
        recommendations["Academic Performance"] = [
            "Schedule a one-on-one tutoring session to address knowledge gaps",
            "Review quiz questions with detailed explanations of correct answers",
            "Create a personalized study plan focusing on weak areas",
            "Provide additional practice materials with immediate feedback"
        ]
        
        if quiz_score < 30:
            recommendations["Academic Performance"].append("Consider fundamental skill assessment to identify prerequisite knowledge gaps")
    
    if "low_video_completion" in issues:
        video_completion = student_data['video_completion_rate']
        recommendations["Content Engagement"] = [
            "Break down longer videos into shorter, focused segments",
            "Add interactive elements to videos to boost engagement",
            "Implement knowledge checkpoints within videos",
            "Provide video summaries and key points as supplementary material"
        ]
        
        if video_completion < 40:
            recommendations["Content Engagement"].append("Check for technical issues affecting video playback")
    
    if "low_engagement" in issues:
        forum_posts = student_data['forum_activity']
        recommendations["Community Engagement"] = [
            "Assign peer collaboration activities",
            "Create discussion prompts related to real-world applications",
            "Recognize and reward active participation",
            "Schedule synchronous discussion sessions"
        ]
        
        if forum_posts < 1:
            recommendations["Community Engagement"].append("Send a personal welcome message to encourage initial participation")
    
    if "high_video_time" in issues:
        avg_time = student_data['avg_time_per_video']
        recommendations["Learning Efficiency"] = [
            "Provide guided notes to focus attention on key concepts",
            "Recommend video playback at 1.25x or 1.5x speed",
            "Suggest the Pomodoro technique (25-minute focused study sessions)",
            "Teach active learning strategies like the Cornell note-taking method"
        ]
        
        if avg_time > 60:
            recommendations["Learning Efficiency"].append("Check if student is pausing/rewatching repeatedly - might indicate confusion")
    
    if "high_location_changes" in issues:
        location_changes = student_data['location_change']
        recommendations["Learning Environment"] = [
            "Recommend creating a dedicated study space",
            "Provide offline access to course materials",
            "Suggest time blocking for focused learning sessions",
            "Share tips for minimizing distractions in different environments"
        ]
    
    # Add motivational strategies for all students
    recommendations["Motivation Strategies"] = [
        "Set up milestone celebrations for course progress",
        "Visualize progress with personalized learning dashboards",
        "Connect course concepts to student's stated career goals",
        "Provide real-world examples relevant to student interests"
    ]
    
    return recommendations

# Function to recommend courses based on quiz scores
def recommend_courses_based_on_quiz(quiz_score):
    """Recommend learning resources based on quiz performance"""
    if quiz_score < 40:  # Struggling students
        return [
            {
                "title": "Foundations: Building Core Knowledge",
                "description": "This course covers the fundamental concepts needed to succeed in more advanced material. Includes interactive exercises and step-by-step tutorials.",
                "link": "https://example.com/foundations"
            },
            {
                "title": "Study Skills Mastery",
                "description": "Learn effective learning strategies, note-taking techniques, and memory methods to improve knowledge retention and test performance.",
                "link": "https://example.com/study-skills"
            },
            {
                "title": "Guided Practice: From Basics to Application",
                "description": "Structured practice sessions with immediate feedback and detailed explanations of common mistakes.",
                "link": "https://example.com/guided-practice"
            }
        ]
    elif quiz_score < 70:  # Intermediate students
        return [
            {
                "title": "Skill Builder: Advancing Your Knowledge",
                "description": "Bridge the gap between foundational concepts and advanced applications with guided practice and case studies.",
                "link": "https://example.com/skill-builder"
            },
            {
                "title": "Problem-Solving Workshop",
                "description": "Enhance your analytical thinking and solution development through structured problem-solving frameworks.",
                "link": "https://example.com/problem-solving"
            },
            {
                "title": "Concept Mastery Series",
                "description": "Deep dives into key concepts with visual explanations, interactive demos, and practical applications.",
                "link": "https://example.com/concept-mastery"
            }
        ]
    else:  # Advanced students
        return [
            {
                "title": "Advanced Applications and Case Studies",
                "description": "Explore complex real-world applications and develop critical analysis skills through challenging case studies.",
                "link": "https://example.com/advanced-applications"
            },
            {
                "title": "Research Methods and Advanced Topics",
                "description": "Learn cutting-edge developments and research methodologies to take your knowledge to expert level.",
                "link": "https://example.com/research-methods"
            },
            {
                "title": "Integration and Synthesis Workshop",
                "description": "Connect concepts across different domains and develop innovative solutions to complex problems.",
                "link": "https://example.com/integration-synthesis"
            }
        ]


def recommend_courses_based_on_quiz(quiz_score):
    """Recommend learning resources based on quiz performance"""
    if quiz_score < 40:  # Struggling students
        return [
            {
                "title": "Foundations: Building Core Knowledge",
                "description": "This course covers the fundamental concepts needed to succeed in more advanced material. Includes interactive exercises and step-by-step tutorials.",
                "link": "https://example.com/foundations"
            },
            {
                "title": "Study Skills Mastery",
                "description": "Learn effective learning strategies, note-taking techniques, and memory methods to improve knowledge retention and test performance.",
                "link": "https://example.com/study-skills"
            },
            {
                "title": "Guided Practice: From Basics to Application",
                "description": "Structured practice sessions with immediate feedback and detailed explanations of common mistakes.",
                "link": "https://example.com/guided-practice"
            }
        ]
    elif quiz_score < 70:  # Intermediate students
        return [
            {
                "title": "Skill Builder: Advancing Your Knowledge",
                "description": "Bridge the gap between foundational concepts and advanced applications with guided practice and case studies.",
                "link": "https://example.com/skill-builder"
            },
            {
                "title": "Problem-Solving Workshop",
                "description": "Enhance your analytical thinking and solution development through structured problem-solving frameworks.",
                "link": "https://example.com/problem-solving"
            },
            {
                "title": "Concept Mastery Series",
                "description": "Deep dives into key concepts with visual explanations, interactive demos, and practical applications.",
                "link": "https://example.com/concept-mastery"
            }
        ]
    else:  # Advanced students
        return [
            {
                "title": "Advanced Applications and Case Studies",
                "description": "Explore complex real-world applications and develop critical analysis skills through challenging case studies.",
                "link": "https://example.com/advanced-applications"
            },
            {
                "title": "Research Methods and Advanced Topics",
                "description": "Learn cutting-edge developments and research methodologies to take your knowledge to expert level.",
                "link": "https://example.com/research-methods"
            },
            {
                "title": "Integration and Synthesis Workshop",
                "description": "Connect concepts across different domains and develop innovative solutions to complex problems.",
                "link": "https://example.com/integration-synthesis"
            }
        ]

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
                    
                    goal_col1, goal_col2 = st.columns(2)
                    
                    with goal_col1:
                        if student_data['quiz_accuracy'] < 80:
                            target_score = st.slider(
                                "Target Quiz Score (%)", 
                                int(student_data['quiz_accuracy']), 
                                100, 
                                int(student_data['quiz_accuracy'] + 15)
                            )
                            score_improvement = target_score - student_data['quiz_accuracy']
                            st.success(f"Target: Improve quiz score by {score_improvement:.1f}% points")
                    
                    with goal_col2:
                        if student_data['video_completion_rate'] < 90:
                            target_completion = st.slider(
                                "Target Video Completion (%)", 
                                int(student_data['video_completion_rate']), 
                                100, 
                                int(min(100, student_data['video_completion_rate'] + 20))
                            )
                            completion_improvement = target_completion - student_data['video_completion_rate']
                            st.success(f"Target: Improve video completion by {completion_improvement:.1f}% points")

# Home Page
if st.session_state.page == 'home':
    st.markdown("<h1 class='main-header'>📚 ISM- E learning Analytics and Recommendation System</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the ISM- E learning Analytics and Recommendation System, a comprehensive solution for improving learning outcomes through 
    data-driven insights and personalized recommendations.
    """)
    
    # Feature cards
    st.markdown("### Platform Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='feature-card anomaly-card'>
            <h3>🔍 Learning Pattern Analysis</h3>
            <p>Identify students who may be struggling or showing unusual learning patterns through advanced analytics.</p>
            <p><b>Key capabilities:</b></p>
            <ul>
                <li>Detect at-risk students early</li>
                <li>Analyze engagement patterns</li>
                <li>Track performance metrics</li>
                <li>Generate personalized recommendations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.button("Anomaly Detection Dashboard", on_click=navigate_to_anomalies, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card recommendation-card'>
            <h3>📚 Course Recommendation Engine</h3>
            <p>Get personalized course recommendations based on learning preferences, interests, and performance data.</p>
            <p><b>Key capabilities:</b></p>
            <ul>
                <li>Content-based filtering</li>
                <li>Preference matching</li>
                <li>Experience-level appropriate suggestions</li>
                <li>Platform and cost optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.button("Course Recommendation Engine", on_click=navigate_to_recommendations, use_container_width=True)
    
    # Dashboard preview image
    st.markdown("### Platform Overview")
    
    # Use a sample dashboard image - replace with your actual dashboard preview
    st.image("https://img.freepik.com/free-vector/education-horizontal-typography-banner-set-with-learning-knowledge-symbols-flat-illustration_1284-29493.jpg", 
             caption="E-Learning Analytics Dashboard")
    
    # Benefits section
    st.markdown("### Benefits")
    
    benefit_col1, benefit_col2, benefit_col3 = st.columns(3)
    
    with benefit_col1:
        st.markdown("""
        #### 👨‍🎓 For Students
        - Personalized learning paths
        - Early intervention when struggling
        - Tailored resource recommendations
        - Progress tracking and goal setting
        """)
    
    with benefit_col2:
        st.markdown("""
        #### 👨‍🏫 For Instructors
        - Identify at-risk students quickly
        - Understand content effectiveness
        - Target interventions efficiently
        - Data-driven course improvements
        """)
    
    with benefit_col3:
        st.markdown("""
        #### 🏫 For Institutions
        - Improve completion rates
        - Enhance student satisfaction
        - Optimize learning resources
        - Scale personalized education
        """)
    
    # Footer
    st.markdown("""
    <div class='footer'>
        <p>© 2025 ISM- E learning Analytics and Recommendation System | Powered by Machine Learning & Data Science</p>
    </div>
    """, unsafe_allow_html=True)

# Anomaly Detection Page
elif st.session_state.page == 'anomalies':
    run_anomaly_detection(reset_to_home)

# Course Recommendation Page 
elif st.session_state.page == 'recommendations':
    run_course_recommendation(reset_to_home)

if __name__ == "__main__":
    # This will run when the script is executed directly
    pass
