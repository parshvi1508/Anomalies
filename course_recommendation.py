import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import joblib

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
    platforms = user_data['Which online Learning Platform due to prefer ?'].dropna().str.strip().unique()
    
    course_data = []
    
    for platform in platforms:
        # Get users who prefer this platform
        platform_users = user_data[user_data['Which online Learning Platform due to prefer ?'] == platform]
        
        # Extract interests for this platform
        interests = platform_users[' Interest'].dropna().str.split(';').explode().str.strip()
        unique_interests = interests.unique()
        
        for interest in unique_interests:
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
                if pd.isna(avg_rating):
                    avg_rating = 2.0  # Default to middle rating if no data
                
                # Extract common difficulty levels
                knowledge_levels = platform_interest_users['Your Knowledge Level: (Select one)'].value_counts().index.tolist()
                difficulty = knowledge_levels[0] if knowledge_levels else "Intermediate (Some Foundational Knowledge)"
                
                # Extract common cost preferences
                cost_prefs = platform_interest_users['Course Cost: (Select your preference)'].value_counts().index.tolist()
                cost_pref = cost_prefs[0] if cost_prefs else "Either, based on value"
                
                # Create a course entry
                course = {
                    'course_id': f"{platform}_{interest.replace(' ', '_')}",
                    'platform': platform,
                    'course_name': f"{interest.strip()} on {platform}",
                    'category': interest.strip(),
                    'skills': interest.strip(),
                    'rating': round(avg_rating, 2),
                    'difficulty_level': difficulty,
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
    """Recommend courses based on user profile
    
    Implementation follows the specification:
    1. Filtering by user preferences
    2. Similarity calculation using TF-IDF
    3. Scoring with weighted combination
    4. Returning top N recommendations
    """
    if course_data.empty:
        return pd.DataFrame()
    
    # Filter courses based on user's preferences
    filtered_courses = course_data.copy()
    
    # If user has specific cost preference (not "Either"), filter by it
    if user_profile['cost_preference'] != "Either, based on value":
        filtered_courses = filtered_courses[
            (filtered_courses['cost_preference'] == user_profile['cost_preference']) | 
            (filtered_courses['cost_preference'] == "Either, based on value")
        ]
    
    # Filter by difficulty level - match user's level
    if user_profile['difficulty'] == "Beginner (No Prior Knowledge)":
        filtered_courses = filtered_courses[
            filtered_courses['difficulty_level'] != "Advanced (Solid Foundational Knowledge)"
        ]
    elif user_profile['difficulty'] == "Advanced (Solid Foundational Knowledge)":
        filtered_courses = filtered_courses[
            filtered_courses['difficulty_level'] != "Beginner (No Prior Knowledge)"
        ]
    
    if filtered_courses.empty:
        # If filtering removed all courses, revert to original dataset
        filtered_courses = course_data.copy()
    
    # Calculate similarity between user interests and course categories
    # Using TF-IDF for text similarity
    
    # Prepare TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    
    # Create corpus of course categories and user domains
    course_categories = filtered_courses['category'].fillna('').tolist()
    domains = [user_profile['domain']]
    corpus = course_categories + domains
    
    # Calculate TF-IDF matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    
    # Calculate cosine similarity between user domains and course categories
    user_vector = tfidf_matrix[-1]  # Last item is the user's domain vector
    course_vectors = tfidf_matrix[:-1]  # All other items are course vectors
    
    # Calculate similarity scores
    similarity_scores = cosine_similarity(user_vector, course_vectors).flatten()
    
    # Add similarity scores to filtered courses
    filtered_courses['similarity'] = similarity_scores
    
    # Calculate final score
    # Weighted combination: 0.6 * similarity + 0.4 * normalized_rating
    rating_weight = 0.4  # Default weight
    similarity_weight = 0.6  # Default weight
    
    # Override with user preferences if available
    if 'importance_of_ratings' in user_profile and 'importance_of_content' in user_profile:
        # Normalize weights to sum to 1
        total = user_profile['importance_of_ratings'] + user_profile['importance_of_content']
        if total > 0:
            rating_weight = user_profile['importance_of_ratings'] / total
            similarity_weight = user_profile['importance_of_content'] / total
    
    # Normalize ratings to 0-1 scale (assuming ratings are 1-5)
    filtered_courses['normalized_rating'] = filtered_courses['rating'] / 3
    
    # Apply weighting
    filtered_courses['score'] = (
        similarity_weight * filtered_courses['similarity'] + 
        rating_weight * filtered_courses['normalized_rating']
    )
    
    # Sort by score and return top N recommendations
    recommendations = filtered_courses.sort_values('score', ascending=False).head(top_n)
    
    return recommendations

def display_recommendations(recommendations, user_profile):
    """Display course recommendations with visualizations"""
    if recommendations.empty:
        st.warning("No suitable courses found based on your preferences. Try adjusting your filters.")
        return
    
    st.subheader("📚 Your Recommended Courses")
    
    # Show each recommendation
    for i, (_, course) in enumerate(recommendations.iterrows()):
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Platform badge/icon
            platform_badge = get_platform_badge(course['platform'])
            st.markdown(platform_badge, unsafe_allow_html=True)
            
            # Rating
            rating_stars = "⭐" * int(round(course['rating']))
            st.markdown(f"**Rating:** {rating_stars} ({course['rating']:.1f})")
            
            # Match score percentage
            match_percent = int(course['similarity'] * 100)
            st.markdown(f"**Match:** {match_percent}%")
        
        with col2:
            st.markdown(f"### {i+1}. {course['course_name']}")
            st.markdown(f"**Category:** {course['category']}")
            st.markdown(f"**Difficulty:** {course['difficulty_level']}")
            st.markdown(f"**Cost:** {course['cost_preference']}")
            
            # Progress bar for total score
            score_percent = int(course['score'] * 100)
            st.markdown(f"**Total Score:** {score_percent}%")
            st.progress(course['score'])
    
    # Show why these courses were recommended
    with st.expander("Why these recommendations?"):
        st.markdown("""
        Courses are recommended based on the following factors:
        
        1. **Content Match**: How well the course content matches your interests
        2. **Rating**: The average rating of the course from other users
        3. **Difficulty Level**: Courses that match your expertise level
        4. **Cost Preference**: Courses that match your cost preference
        
        The final score is a weighted combination of these factors, with content match
        and rating being the most important.
        """)
    
    # Visualization of recommendations
    st.subheader("Recommendation Analysis")
    
    # Bar chart comparing different courses
    fig = px.bar(
        recommendations,
        y="course_name",
        x=["similarity", "normalized_rating"],
        labels={"value": "Score", "variable": "Factor"},
        title="Recommendation Factors by Course",
        orientation='h',
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Radar chart showing how different platforms meet needs
    platform_data = recommendations.groupby('platform').agg({
        'similarity': 'mean',
        'normalized_rating': 'mean',
        'score': 'mean',
        'course_name': 'count'
    }).reset_index()
    platform_data.rename(columns={'course_name': 'course_count'}, inplace=True)
    
    if len(platform_data) > 1:
        fig = px.line_polar(
            platform_data, 
            r="score", 
            theta="platform", 
            line_close=True,
            range_r=[0,1],
            title="Platform Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)

def get_platform_badge(platform):
    """Return HTML for a platform badge"""
    platform = platform.lower()
    color = "#3366cc"  # Default blue
    
    if "coursera" in platform:
        color = "#0056D2"
    elif "udemy" in platform:
        color = "#A435F0"
    elif "linkedin" in platform:
        color = "#0077B5"
    elif "edx" in platform:
        color = "#01262F"
    elif "youtube" in platform:
        color = "#FF0000"
    
    return f'<div style="background-color: {color}; color: white; padding: 8px; border-radius: 4px; text-align: center; font-weight: bold;">{platform}</div>'

def show_data_exploration(user_data, course_data):
    """Show data exploration visualizations"""
    with st.expander("📊 Data Exploration", expanded=False):
        if user_data is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                # Platform popularity
                platform_counts = user_data['Which online Learning Platform due to prefer ?'].value_counts().reset_index()
                platform_counts.columns = ['Platform', 'Count']
                
                fig = px.bar(
                    platform_counts.head(10), 
                    x='Platform', 
                    y='Count',
                    title="Most Popular Learning Platforms"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Interest areas
                interests = user_data[' Interest'].dropna().str.split(';').explode().str.strip()
                interest_counts = interests.value_counts().reset_index()
                interest_counts.columns = ['Interest', 'Count']
                
                fig = px.pie(
                    interest_counts.head(5), 
                    values='Count', 
                    names='Interest',
                    title="Top Interest Areas"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        if not course_data.empty:
            # Course data exploration
            col1, col2 = st.columns(2)
            
            with col1:
                # Ratings by platform
                avg_ratings = course_data.groupby('platform')['rating'].mean().reset_index()
                
                fig = px.bar(
                    avg_ratings, 
                    x='platform', 
                    y='rating',
                    title="Average Rating by Platform",
                    color='rating',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Categories by popularity
                category_counts = course_data['category'].value_counts().reset_index()
                category_counts.columns = ['Category', 'Count']
                
                fig = px.bar(
                    category_counts.head(10), 
                    x='Count', 
                    y='Category',
                    title="Most Popular Categories",
                    orientation='h'
                )
                st.plotly_chart(fig, use_container_width=True)