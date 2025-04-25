import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import re

def run_course_recommendation(reset_callback):
    """Run the course recommendation section of the app"""
    st.title("📚 Course Recommendation System")
    st.button("← Back to Home", on_click=reset_callback)
    
    st.markdown("""
        Get personalized course recommendations based on your preferences and interests.
        Upload a course catalog or use our sample data to begin.
    """)
    
    # Load the user survey data
    user_data = load_user_data()
    
    if user_data is not None:
        # Create course data from survey responses
        course_data = extract_course_data(user_data)
        
        # Input method selection
        method = st.radio("Select input method:", ["Enter Your Information", "Use Sample Profile"])
        
        if method == "Enter Your Information":
            user_profile = get_user_input()
            
            if st.button("Generate Recommendations"):
                if user_profile:
                    recommendations = recommend_courses_for_user(user_profile, course_data)
                    display_recommendations(recommendations, user_profile)
                else:
                    st.warning("Please fill in your information to get recommendations.")
                    
        else:  # Use Sample Profile
            sample_profile = get_sample_profile()
            st.subheader("Sample User Profile")
            display_user_profile(sample_profile)
            
            if st.button("Generate Recommendations"):
                recommendations = recommend_courses_for_user(sample_profile, course_data)
                display_recommendations(recommendations, sample_profile)
                
        # Show data exploration section
        show_data_exploration(user_data, course_data)

def load_user_data():
    """Load the user survey data from CSV"""
    try:
        df = pd.read_csv('Course_Recommender_500_Users.csv')
        return df
    except Exception as e:
        st.error(f"Error loading user survey data: {str(e)}")
        st.info("You can upload your own CSV file with user data:")
        uploaded_file = st.file_uploader("Upload User Data CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                return df
            except Exception as e:
                st.error(f"Error loading uploaded file: {str(e)}")
        return None

def extract_course_data(user_data):
    """Extract course information from user survey data"""
    # For each platform, gather the interests, rating, difficulty level, etc.
    platforms = user_data['Which online Learning Platform due to prefer ?'].dropna().str.split(';').explode().str.strip().unique()
    
    course_data = []
    
    for platform in platforms:
        # Get users who prefer this platform
        platform_users = user_data[user_data['Which online Learning Platform due to prefer ?'].str.contains(platform, na=False)]
        
        # Extract interests for this platform
        interests = platform_users[' Interest'].dropna().str.split(';').explode().str.strip().unique()
        
        for interest in interests:
            # Calculate average engagement and satisfaction for this platform and interest
            platform_interest_users = platform_users[platform_users[' Interest'].str.contains(interest, na=False)]
            
            if len(platform_interest_users) > 0:
                # Convert engagement and satisfaction to numerical values
                engagement_map = {
                    'Not Engaging': 1, 'Somewhat Engaging': 2, 'Very Engaging': 3
                }
                satisfaction_map = {
                    'Not Satisfied': 1, 'Somewhat Satisfied': 2, 'Very Satisfied': 3
                }
                
                # Map text values to numerical scores with error handling
                engagement_scores = platform_interest_users['How engaging and interesting did you find the course content (presentations, videos, activities)? '].apply(
                    lambda x: engagement_map.get(x, 0) if pd.notna(x) else 0
                )
                satisfaction_scores = platform_interest_users['How satisfied were you with the learning experience on this platform? '].apply(
                    lambda x: satisfaction_map.get(x, 0) if pd.notna(x) else 0
                )
                
                avg_rating = (engagement_scores.mean() + satisfaction_scores.mean()) / 2
                
                # Extract common difficulty levels
                knowledge_levels = platform_interest_users['Your Knowledge Level: (Select one)'].value_counts().index.tolist()
                difficulty = knowledge_levels[0] if knowledge_levels else "Intermediate (Some Foundational Knowledge)"
                
                # Extract common cost preferences
                cost_prefs = platform_interest_users['Course Cost: (Select your preference)'].value_counts().index.tolist()
                cost_pref = cost_prefs[0] if cost_prefs else "Either, based on value"
                
                # Create a course entry
                course = {
                    'platform': platform,
                    'title': f"{interest.strip()} on {platform}",
                    'domain': interest.strip(),
                    'rating': round(avg_rating, 2),
                    'difficulty': difficulty,
                    'cost_preference': cost_pref,
                    'user_count': len(platform_interest_users)
                }
                
                course_data.append(course)
    
    return pd.DataFrame(course_data)

def get_user_input():
    """Get user preferences for recommendation"""
    st.subheader("Your Interests and Preferences")
    
    # Domain of interest
    domain_options = [
        "Programming Languages", 
        "Data Science", 
        "Business", 
        "Design", 
        "Marketing", 
        "Soft Skills"
    ]
    domain = st.multiselect("Select your domains of interest:", domain_options)
    
    # Experience level
    difficulty_options = [
        "Beginner (No Prior Knowledge)",
        "Intermediate (Some Foundational Knowledge)",
        "Advanced (Solid Foundational Knowledge)"
    ]
    difficulty = st.selectbox("Your experience level:", difficulty_options)
    
    # Cost preference
    cost_options = [
        "Free", 
        "Paid", 
        "Either, based on value"
    ]
    cost_pref = st.selectbox("Your cost preference:", cost_options)
    
    # Additional preferences
    col1, col2 = st.columns(2)
    with col1:
        importance_of_ratings = st.slider("Importance of course ratings:", 1, 10, 7)
    with col2:
        importance_of_content = st.slider("Importance of content match:", 1, 10, 8)
    
    if not domain:
        st.warning("Please select at least one domain of interest.")
        return None
    
    # Create user profile
    user_profile = {
        'domain': "; ".join(domain),
        'difficulty': difficulty,
        'cost_preference': cost_pref,
        'importance_of_ratings': importance_of_ratings / 10,  # Normalize to 0-1
        'importance_of_content': importance_of_content / 10   # Normalize to 0-1
    }
    
    return user_profile

def get_sample_profile():
    """Return a sample user profile for demonstration"""
    return {
        'domain': "Data Science; Programming Languages",
        'difficulty': "Intermediate (Some Foundational Knowledge)",
        'cost_preference': "Either, based on value",
        'importance_of_ratings': 0.7,
        'importance_of_content': 0.8
    }

def display_user_profile(profile):
    """Display user profile information"""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Domains of Interest:** {profile['domain']}")
        st.markdown(f"**Experience Level:** {profile['difficulty']}")
    with col2:
        st.markdown(f"**Cost Preference:** {profile['cost_preference']}")
        st.markdown(f"**Rating Importance:** {profile['importance_of_ratings']:.1f}")
        st.markdown(f"**Content Match Importance:** {profile['importance_of_content']:.1f}")

def recommend_courses_for_user(user_profile, course_data, top_n=5):
    """Recommend courses based on user profile"""
    if course_data.empty:
        return pd.DataFrame()
    
    # Filter courses based on difficulty level and cost preference
    filtered_courses = course_data.copy()
    
    # If user has specific cost preference (not "Either"), filter by it
    if user_profile['cost_preference'] != "Either, based on value":
        filtered_courses = filtered_courses[
            (filtered_courses['cost_preference'] == user_profile['cost_preference']) | 
            (filtered_courses['cost_preference'] == "Either, based on value")
        ]
    
    # If user is beginner, filter out advanced courses
    if user_profile['difficulty'] == "Beginner (No Prior Knowledge)":
        filtered_courses = filtered_courses[
            filtered_courses['difficulty'] != "Advanced (Solid Foundational Knowledge)"
        ]
    
    # If user is advanced, prefer advanced courses
    elif user_profile['difficulty'] == "Advanced (Solid Foundational Knowledge)":
        # Boost the score for advanced courses later
        pass
    
    if filtered_courses.empty:
        # If filtering removed all courses, revert to original dataset
        filtered_courses = course_data.copy()
    
    # Calculate similarity between user interests and course domains
    # Using TF-IDF and cosine similarity
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    
    # Combine all domains and user interests for TF-IDF
    all_domains = list(filtered_courses['domain'].values) + [user_profile['domain']]
    
    # Handle empty values
    all_domains = [d if isinstance(d, str) else "" for d in all_domains]
    
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_domains)
    
    # Get the user's interests vector (last item in the matrix)
    user_interests_vector = tfidf_matrix[-1]
    
    # Get the courses' domain vectors
    courses_vectors = tfidf_matrix[:-1]
    
    # Calculate cosine similarity between user interests and course domains
    similarity_scores = cosine_similarity(user_interests_vector, courses_vectors)
    
    # Add similarity scores to filtered courses
    filtered_courses['similarity'] = similarity_scores[0]
    
    # Calculate final score
    # Weight based on user's preferences
    rating_weight = user_profile['importance_of_ratings']
    similarity_weight = user_profile['importance_of_content']
    
    # Normalize weights
    total_weight = rating_weight + similarity_weight
    rating_weight = rating_weight / total_weight
    similarity_weight = similarity_weight / total_weight
    
    # Add difficulty bonus for matching experience level
    filtered_courses['difficulty_bonus'] = 0.0
    filtered_courses.loc[filtered_courses['difficulty'] == user_profile['difficulty'], 'difficulty_bonus'] = 0.1
    
    # Calculate final score
    filtered_courses['score'] = (
        similarity_weight * filtered_courses['similarity'] +
        rating_weight * (filtered_courses['rating'] / 3) +  # Normalize ratings to 0-1 scale
        filtered_courses['difficulty_bonus']
    )
    
    # Sort by score
    recommendations = filtered_courses.sort_values('score', ascending=False).head(top_n)
    
    return recommendations

def display_recommendations(recommendations, user_profile):
    """Display course recommendations with visualizations"""
    if recommendations.empty:
        st.warning("No suitable courses found based on your preferences. Try adjusting your filters.")
        return
    
    st.subheader("📚 Your Recommended Courses")
    
    # Show the top courses as cards
    for idx, course in recommendations.iterrows():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Platform icon or logo
            platform_icon = get_platform_icon(course['platform'])
            st.markdown(platform_icon, unsafe_allow_html=True)
            
            # Rating stars
            stars = "⭐" * int(round(course['rating']))
            st.markdown(f"**Rating:** {stars} ({course['rating']:.1f})")
        
        with col2:
            st.markdown(f"### {course['title']}")
            st.markdown(f"**Domain:** {course['domain']}")
            st.markdown(f"**Difficulty:** {course['difficulty']}")
            st.markdown(f"**Platform:** {course['platform']}")
            
            # Match score as a progress bar
            match_percentage = int(course['similarity'] * 100)
            st.markdown(f"**Content Match:** {match_percentage}%")
            st.progress(course['similarity'])
    
    # Visualization of recommendations
    st.subheader("Recommendation Analysis")
    
    # Create a radar chart of the top recommendations
    fig = px.line_polar(
        recommendations, 
        r='score', 
        theta='domain',
        line_close=True,
        range_r=[0, 1],
        title="Course Match Analysis"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Bar chart comparing the recommendations
    fig = px.bar(
        recommendations,
        x='title',
        y=['similarity', 'rating'/3],  # Normalize rating to 0-1 scale
        title="Factor Comparison Across Recommendations",
        labels={
            'value': 'Score', 
            'variable': 'Factor',
            'title': 'Course'
        },
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

def get_platform_icon(platform):
    """Return an HTML icon for the platform"""
    platform = platform.lower()
    if 'coursera' in platform:
        return '<div style="font-size:42px; color:#0056D2;">🔷</div>'
    elif 'udemy' in platform:
        return '<div style="font-size:42px; color:#A435F0;">🟣</div>'
    elif 'linkedin' in platform:
        return '<div style="font-size:42px; color:#0077B5;">📘</div>'
    elif 'nptel' in platform:
        return '<div style="font-size:42px; color:#F47C48;">📙</div>'
    elif 'youtube' in platform:
        return '<div style="font-size:42px; color:#FF0000;">▶️</div>'
    else:
        return '<div style="font-size:42px; color:#3E4042;">🎓</div>'

def show_data_exploration(user_data, course_data):
    """Show data exploration section"""
    with st.expander("📊 Data Exploration", expanded=False):
        st.subheader("Platform Analysis")
        
        # Platform popularity
        platforms = user_data['Which online Learning Platform due to prefer ?'].dropna().str.split(';').explode().str.strip()
        platform_counts = platforms.value_counts().reset_index()
        platform_counts.columns = ['Platform', 'Count']
        
        fig = px.bar(
            platform_counts, 
            x='Platform', 
            y='Count',
            title="Popular Learning Platforms",
            color='Count',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Interest distribution
        interests = user_data[' Interest'].dropna().str.split(';').explode().str.strip()
        interest_counts = interests.value_counts().reset_index()
        interest_counts.columns = ['Interest', 'Count']
        
        fig = px.pie(
            interest_counts.head(6), 
            values='Count', 
            names='Interest',
            title="Popular Interest Areas"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Course analysis
        if not course_data.empty:
            st.subheader("Course Analysis")
            
            # Rating distribution
            fig = px.histogram(
                course_data,
                x='rating',
                nbins=10,
                title="Course Rating Distribution",
                labels={'rating': 'Rating'}
            )
            st.plotly_chart(fig, use_container_width=True)