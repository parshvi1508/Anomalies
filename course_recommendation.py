'''
# course_recommendations.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st
import re
import os

# Constants
SIMILARITY_WEIGHT = 0.6
RATING_WEIGHT = 0.4

def load_course_data(file_path='udemy_courses.csv'):
    """Load and preprocess course data"""
    courses_df = pd.read_csv(file_path)
    courses_df = courses_df.fillna('')
    
    # Create a skills field from course_title and subject
    courses_df['skills'] = courses_df['course_title'] + ' ' + courses_df['subject']
    
    # Convert price to numeric
    courses_df['price'] = pd.to_numeric(courses_df['price'], errors='coerce')
    
    # Calculate rating proxy using num_reviews and num_subscribers
    if 'num_reviews' in courses_df.columns and 'num_subscribers' in courses_df.columns:
        courses_df['normalized_reviews'] = normalize_column(courses_df, 'num_reviews')
        courses_df['normalized_subscribers'] = normalize_column(courses_df, 'num_subscribers')
        
        # Combine normalized reviews and subscribers as a proxy for rating
        courses_df['normalized_rating'] = (
            0.7 * courses_df['normalized_reviews'] + 
            0.3 * courses_df['normalized_subscribers']
        )
    else:
        courses_df['normalized_rating'] = 0.5  # Default if no ratings
    
    # Map cost and knowledge level
    courses_df['cost_type'] = courses_df['is_paid'].apply(lambda x: 'Paid' if x == True else 'Free')
    
    level_mapping = {
        'Beginner Level': 'Beginner',
        'Intermediate Level': 'Intermediate',
        'Expert Level': 'Advanced',
        'All Levels': 'All Levels'
    }
    courses_df['knowledge_level'] = courses_df['level'].map(level_mapping).fillna('All Levels')
    
    return courses_df

def normalize_column(df, column_name):
    """Normalize a column to a 0-1 scale"""
    max_val = df[column_name].max()
    min_val = df[column_name].min()
    if max_val > min_val:
        return (df[column_name] - min_val) / (max_val - min_val)
    else:
        return pd.Series(0.5, index=df.index)

def preprocess_text(text):
    """Preprocess text for better TF-IDF results"""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_tfidf_vectorizer(courses_df):
    """Create and fit a TF-IDF vectorizer on course skills"""
    course_skills = courses_df['skills'].apply(preprocess_text).tolist()
    
    # Create and fit the TF-IDF vectorizer
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.0, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(course_skills)
    
    return tfidf, tfidf_matrix

def recommend_courses_for_user(user_profile, courses_df, tfidf_vectorizer, tfidf_matrix, top_n=5):
    """Recommend courses for a user based on their profile"""
    # Extract user preferences
    domain_of_interest = user_profile.get('domain_of_interest', '')
    cost_preference = user_profile.get('cost_preference', 'Free')
    knowledge_level = user_profile.get('knowledge_level', 'Beginner')
    
    # Preprocess domain of interest
    domain_of_interest = preprocess_text(domain_of_interest)
    
    # Filter courses based on cost preference
    if cost_preference == 'Free':
        filtered_courses = courses_df[courses_df['cost_type'] == 'Free']
    elif cost_preference == 'Paid':
        filtered_courses = courses_df[courses_df['cost_type'] == 'Paid']
    else:
        filtered_courses = courses_df
    
    # Filter courses based on knowledge level
    if knowledge_level == 'Beginner':
        level_filter = filtered_courses['knowledge_level'].isin(['Beginner', 'All Levels'])
    elif knowledge_level == 'Intermediate':
        level_filter = filtered_courses['knowledge_level'].isin(['Intermediate', 'All Levels'])
    elif knowledge_level == 'Advanced':
        level_filter = filtered_courses['knowledge_level'].isin(['Advanced', 'All Levels'])
    else:
        level_filter = True
    
    filtered_courses = filtered_courses[level_filter]
    
    # If no courses match the filters, return empty DataFrame
    if filtered_courses.empty:
        return pd.DataFrame()
    
    # Vectorize user's domain of interest
    user_vector = tfidf_vectorizer.transform([domain_of_interest])
    
    # Calculate similarity scores between user's interest and filtered courses
    indices = filtered_courses.index.tolist()
    
    # Using linear_kernel for cosine similarity (faster than cosine_similarity for TF-IDF)
    similarity_scores = linear_kernel(user_vector, tfidf_matrix[indices]).flatten()
    
    # Create a DataFrame with filtered courses and their similarity scores
    recommendations = filtered_courses.copy()
    recommendations['similarity_score'] = similarity_scores
    
    # Calculate final score combining similarity and rating
    recommendations['final_score'] = (
        SIMILARITY_WEIGHT * recommendations['similarity_score'] + 
        RATING_WEIGHT * recommendations['normalized_rating']
    )
    
    # Sort by final score and return top N recommendations
    recommendations = recommendations.sort_values('final_score', ascending=False).head(top_n)
    
    return recommendations[['course_id', 'course_title', 'level', 'cost_type', 'price', 
                          'similarity_score', 'final_score', 'url']]

def parse_user_survey_data(survey_row):
    """Parse user data from survey response"""
    user_profile = {
        'domain_of_interest': '',
        'cost_preference': 'Free',
        'knowledge_level': 'Beginner'
    }
    
    # Extract domain of interest
    domain_field = 'Domain of Interest: (Select all that apply)'
    if domain_field in survey_row and pd.notna(survey_row[domain_field]):
        domain_interests = []
        interests = str(survey_row[domain_field]).split(';')
        for interest in interests:
            main_category = interest.split('(')[0].strip()
            domain_interests.append(main_category)
        
        user_profile['domain_of_interest'] = ' '.join(domain_interests)
    
    # Extract cost preference
    cost_field = 'Course Cost: (Select your preference)'
    if cost_field in survey_row and pd.notna(survey_row[cost_field]):
        if 'Free' in str(survey_row[cost_field]):
            user_profile['cost_preference'] = 'Free'
        elif 'Paid' in str(survey_row[cost_field]):
            user_profile['cost_preference'] = 'Paid'
    
    # Extract knowledge level
    knowledge_field = 'Your Knowledge Level: (Select one)'
    if knowledge_field in survey_row and pd.notna(survey_row[knowledge_field]):
        if 'Beginner' in str(survey_row[knowledge_field]):
            user_profile['knowledge_level'] = 'Beginner'
        elif 'Intermediate' in str(survey_row[knowledge_field]):
            user_profile['knowledge_level'] = 'Intermediate'
        elif 'Advanced' in str(survey_row[knowledge_field]):
            user_profile['knowledge_level'] = 'Advanced'
    
    return user_profile

class CourseRecommender:
    """A class to handle course recommendations"""
    def __init__(self, course_data_path='udemy_courses.csv'):
        """Initialize the recommender"""
        self.courses_df = load_course_data(course_data_path)
        self.tfidf_vectorizer, self.tfidf_matrix = create_tfidf_vectorizer(self.courses_df)
    
    def recommend_courses(self, user_profile, top_n=5):
        """Recommend courses for a user"""
        return recommend_courses_for_user(
            user_profile, 
            self.courses_df, 
            self.tfidf_vectorizer, 
            self.tfidf_matrix, 
            top_n
        )
    
    def recommend_courses_from_survey(self, survey_row, top_n=5):
        """Recommend courses based on survey response"""
        user_profile = parse_user_survey_data(survey_row)
        return self.recommend_courses(user_profile, top_n)

def extract_knowledge_domains(survey_df):
    """Extract unique knowledge domains from survey data"""
    domains = set()
    domain_field = 'Domain of Interest: (Select all that apply)'
    
    if domain_field in survey_df.columns:
        for interests in survey_df[domain_field].dropna():
            for interest in str(interests).split(';'):
                domain = interest.split('(')[0].strip()
                if domain:
                    domains.add(domain)
    
    return sorted(list(domains))

def run_course_recommendation(reset_callback):
    """Run the course recommendation section of the app"""
    st.markdown("<h1 class='sub-header'>📚 Course Recommendation Engine</h1>", unsafe_allow_html=True)
    st.button("← Back to Home", on_click=reset_callback)
    
    st.markdown("""
        This recommendation engine suggests courses based on your interests and preferences.
        Fill out the form below to get personalized course recommendations.
    """)
    
    try:
        # Check if survey data exists and load it
        survey_df = None
        if os.path.exists('Course-Recommender-System.csv'):
            survey_df = pd.read_csv('Course-Recommender-System.csv')
        
        # Initialize the recommender with courses
        recommender = CourseRecommender()
        
        # Dashboard layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("<h3>🧑‍💻 Your Preferences</h3>", unsafe_allow_html=True)
            
            # User profile input form
            with st.form("user_profile_form"):
                # Domains of interest
                if survey_df is not None:
                    domains = extract_knowledge_domains(survey_df)
                    domain_options = domains if domains else ["Programming", "Data Science", "Business", "Design", "Marketing"]
                else:
                    domain_options = ["Programming", "Data Science", "Business", "Design", "Marketing"]
                
                selected_domains = st.multiselect(
                    "Select your domains of interest:",
                    options=domain_options,
                    default=domain_options[:1] if domain_options else []
                )
                
                # Cost preference
                cost_preference = st.selectbox(
                    "Course cost preference:",
                    options=["Any", "Free", "Paid"],
                    index=0
                )
                
                # Knowledge level
                knowledge_level = st.selectbox(
                    "Your knowledge level:",
                    options=["Beginner", "Intermediate", "Advanced", "All Levels"],
                    index=0
                )
                
                # Number of recommendations
                num_recommendations = st.slider(
                    "Number of recommendations:",
                    min_value=3,
                    max_value=10,
                    value=5
                )
                
                # Additional preferences (optional)
                with st.expander("Additional Preferences (Optional)"):
                    max_duration = st.slider(
                        "Maximum course duration (hours):",
                        min_value=1,
                        max_value=20,
                        value=10
                    )
                    
                    min_rating = st.slider(
                        "Minimum course rating:",
                        min_value=1.0,
                        max_value=5.0,
                        value=4.0,
                        step=0.1
                    )
                
                # Submit button
                submitted = st.form_submit_button("Get Recommendations")
            
            # User profile from survey data
            if survey_df is not None:
                st.markdown("<h3>📊 or Select from Survey</h3>", unsafe_allow_html=True)
                
                if not survey_df.empty:
                    # Get unique usernames
                    usernames = survey_df['Username'].dropna().unique()
                    
                    if len(usernames) > 0:
                        selected_user = st.selectbox(
                            "Select a user from survey data:",
                            options=usernames
                        )
                        
                        use_survey_data = st.button("Use This User's Preferences")
                    else:
                        st.warning("No usernames found in survey data")
                        use_survey_data = False
                else:
                    st.warning("Survey data is empty")
                    use_survey_data = False
        
        with col2:
            st.markdown("<h3>🎓 Recommended Courses</h3>", unsafe_allow_html=True)
            
            # Process recommendations based on form submission
            if submitted:
                if not selected_domains:
                    st.error("Please select at least one domain of interest")
                else:
                    # Create user profile
                    user_profile = {
                        'domain_of_interest': ' '.join(selected_domains),
                        'cost_preference': cost_preference,
                        'knowledge_level': knowledge_level
                    }
                    
                    with st.spinner('Finding the best courses for you...'):
                        # Get recommendations
                        recommendations = recommender.recommend_courses(user_profile, num_recommendations)
                        
                        if recommendations.empty:
                            st.warning("No courses match your preferences. Try adjusting your filters.")
                        else:
                            # Display recommendations
                            for i, (_, course) in enumerate(recommendations.iterrows()):
                                with st.container():
                                    st.markdown(f"""
                                    <div class='feature-card recommendation-card'>
                                        <h4>{i+1}. {course['course_title']}</h4>
                                        <p><b>Level:</b> {course['level']} | <b>Cost:</b> {course['cost_type']} {f"(${course['price']:.2f})" if course['cost_type'] == 'Paid' else ''}</p>
                                        <p><b>Match Score:</b> {course['similarity_score']*100:.1f}% | <b>Overall Score:</b> {course['final_score']*100:.1f}%</p>
                                        <p><a href="{course['url']}" target="_blank">View Course</a></p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    st.markdown("---")
            
            # Process recommendations based on survey selection
            elif 'use_survey_data' in locals() and use_survey_data and 'selected_user' in locals():
                with st.spinner('Finding the best courses based on survey data...'):
                    # Get the selected user's data
                    user_row = survey_df[survey_df['Username'] == selected_user].iloc[0]
                    
                    # Get recommendations from survey data
                    recommendations = recommender.recommend_courses_from_survey(user_row, num_recommendations)
                    
                    if recommendations.empty:
                        st.warning("No courses match this user's preferences. Try another user or adjust filters manually.")
                    else:
                        # Display user preferences from survey
                        user_profile = parse_user_survey_data(user_row)
                        
                        st.markdown("<h4>User's Preferences:</h4>", unsafe_allow_html=True)
                        st.markdown(f"**Domains of Interest:** {user_profile['domain_of_interest']}")
                        st.markdown(f"**Cost Preference:** {user_profile['cost_preference']}")
                        st.markdown(f"**Knowledge Level:** {user_profile['knowledge_level']}")
                        st.markdown("---")
                        
                        # Display recommendations
                        for i, (_, course) in enumerate(recommendations.iterrows()):
                            with st.container():
                                st.markdown(f"""
                                <div class='feature-card recommendation-card'>
                                    <h4>{i+1}. {course['course_title']}</h4>
                                    <p><b>Level:</b> {course['level']} | <b>Cost:</b> {course['cost_type']} {f"(${course['price']:.2f})" if course['cost_type'] == 'Paid' else ''}</p>
                                    <p><b>Match Score:</b> {course['similarity_score']*100:.1f}% | <b>Overall Score:</b> {course['final_score']*100:.1f}%</p>
                                    <p><a href="{course['url']}" target="_blank">View Course</a></p>
                                </div>
                                """, unsafe_allow_html=True)
                                st.markdown("---")
            else:
                # Display placeholder when no recommendations yet
                st.info("Fill out the form and click 'Get Recommendations' to see personalized course suggestions.")
                
                # Alternative courses display for exploration
                with st.expander("Browse Popular Courses", expanded=False):
                    # Simply show top-rated courses as a fallback
                    popular_courses = recommender.courses_df.sort_values('normalized_rating', ascending=False).head(5)
                    for i, (_, course) in enumerate(popular_courses.iterrows()):
                        st.markdown(f"**{i+1}. {course['course_title']}**")
                        st.markdown(f"Level: {course['level']} | Cost: {course['cost_type']}")
                        st.markdown(f"[View Course]({course['url']})")
                        st.markdown("---")
        
        # Analytics section
        with st.expander("📊 Recommendation Analytics", expanded=False):
            st.markdown("### Insights from Course Data")
            
            # Calculate analytics
            courses_df = recommender.courses_df
            
            # Distribution of courses by level
            level_counts = courses_df['knowledge_level'].value_counts().reset_index()
            level_counts.columns = ['Level', 'Count']
            
            # Distribution of courses by cost type
            cost_counts = courses_df['cost_type'].value_counts().reset_index()
            cost_counts.columns = ['Cost Type', 'Count']
            
            # Layout for charts
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.subheader("Courses by Level")
                st.bar_chart(level_counts.set_index('Level'))
            
            with chart_col2:
                st.subheader("Courses by Cost")
                st.bar_chart(cost_counts.set_index('Cost Type'))
            
            # Course subjects analysis
            if 'subject' in courses_df.columns:
                subject_counts = courses_df['subject'].value_counts().head(10).reset_index()
                subject_counts.columns = ['Subject', 'Count']
                
                st.subheader("Top Course Subjects")
                st.bar_chart(subject_counts.set_index('Subject'))
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

if __name__ == "__main__":
    # This allows running this file directly for testing
    st.set_page_config(page_title="Course Recommender", layout="wide")
    
    # Placeholder for reset callback
    def temp_reset():
        pass
    
    # Run the recommendation page
    run_course_recommendation(temp_reset)
'''
# course_recommendations.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st
import re
import os
from datetime import datetime

# Constants
SIMILARITY_WEIGHT = 0.6
RATING_WEIGHT = 0.4

def load_course_data(file_path='coursera_1000.csv'):
    """Load and preprocess course data"""
    try:
        courses_df = pd.read_csv(file_path)
        courses_df = courses_df.fillna('')
        
        # Create a skills field from course_title and subject
        courses_df['skills'] = courses_df['course_title'] + ' ' + courses_df['subject']
        
        # Convert price to numeric
        courses_df['price'] = pd.to_numeric(courses_df['price'], errors='coerce')
        
        # Calculate rating proxy using num_reviews and num_subscribers
        if 'num_reviews' in courses_df.columns and 'num_subscribers' in courses_df.columns:
            courses_df['normalized_reviews'] = normalize_column(courses_df, 'num_reviews')
            courses_df['normalized_subscribers'] = normalize_column(courses_df, 'num_subscribers')
            
            # Combine normalized reviews and subscribers as a proxy for rating
            courses_df['normalized_rating'] = (
                0.7 * courses_df['normalized_reviews'] + 
                0.3 * courses_df['normalized_subscribers']
            )
        else:
            courses_df['normalized_rating'] = 0.5  # Default if no ratings
        
        # Map cost and knowledge level
        if 'is_paid' in courses_df.columns:
            courses_df['cost_type'] = courses_df['is_paid'].apply(lambda x: 'Paid' if x == True else 'Free')
        else:
            courses_df['cost_type'] = 'Unknown'
        
        level_mapping = {
            'Beginner Level': 'Beginner',
            'Intermediate Level': 'Intermediate',
            'Expert Level': 'Advanced',
            'All Levels': 'All Levels'
        }
        
        if 'level' in courses_df.columns:
            courses_df['knowledge_level'] = courses_df['level'].map(level_mapping).fillna('All Levels')
        else:
            courses_df['knowledge_level'] = 'All Levels'
        
        return courses_df
    except Exception as e:
        st.error(f"Error loading course data: {str(e)}")
        return pd.DataFrame()

def normalize_column(df, column_name):
    """Normalize a column to a 0-1 scale"""
    max_val = df[column_name].max()
    min_val = df[column_name].min()
    if max_val > min_val:
        return (df[column_name] - min_val) / (max_val - min_val)
    else:
        return pd.Series(0.5, index=df.index)

def preprocess_text(text):
    """Preprocess text for better TF-IDF results"""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_tfidf_vectorizer(courses_df):
    """Create and fit a TF-IDF vectorizer on course skills"""
    course_skills = courses_df['skills'].apply(preprocess_text).tolist()
    
    # Create and fit the TF-IDF vectorizer
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.0, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(course_skills)
    
    return tfidf, tfidf_matrix

def train_random_forest_model(user_data, course_data):
    """Train a Random Forest model for course recommendations"""
    # Prepare training data
    # We'll use user preferences and course features to predict user satisfaction
    
    # First, let's create a synthetic dataset based on user preferences
    training_data = []
    
    # Extract unique domains of interest
    domains = set()
    for _, user in user_data.iterrows():
        if 'Domain of Interest: (Select all that apply)' in user:
            interests = str(user['Domain of Interest: (Select all that apply)']).split(';')
            for interest in interests:
                domain = interest.split('(')[0].strip()
                if domain:
                    domains.add(domain)
    
    # Create encoders for categorical features
    domain_encoder = LabelEncoder()
    domain_encoder.fit(list(domains))
    
    level_encoder = LabelEncoder()
    level_encoder.fit(['Beginner', 'Intermediate', 'Advanced', 'All Levels'])
    
    cost_encoder = LabelEncoder()
    cost_encoder.fit(['Free', 'Paid'])
    
    # Generate synthetic training data
    for _, user in user_data.iterrows():
        user_level = 'Beginner'
        if 'Your Knowledge Level: (Select one)' in user:
            if 'Intermediate' in str(user['Your Knowledge Level: (Select one)']):
                user_level = 'Intermediate'
            elif 'Advanced' in str(user['Your Knowledge Level: (Select one)']):
                user_level = 'Advanced'
        
        user_cost = 'Free'
        if 'Course Cost: (Select your preference)' in user:
            if 'Paid' in str(user['Course Cost: (Select your preference)']):
                user_cost = 'Paid'
        
        # For each user, generate positive and negative examples
        for _, course in course_data.iterrows():
            # Features
            course_level = course['knowledge_level']
            course_cost = course['cost_type']
            
            # Calculate level match (0-1)
            if user_level == 'Beginner' and course_level in ['Beginner', 'All Levels']:
                level_match = 1.0
            elif user_level == 'Intermediate' and course_level in ['Intermediate', 'All Levels']:
                level_match = 1.0
            elif user_level == 'Advanced' and course_level in ['Advanced', 'All Levels']:
                level_match = 1.0
            else:
                level_match = 0.0
            
            # Calculate cost match (0-1)
            cost_match = 1.0 if user_cost == course_cost else 0.0
            
            # Calculate domain match (0-1)
            domain_match = 0.0
            if 'Domain of Interest: (Select all that apply)' in user:
                user_domains = str(user['Domain of Interest: (Select all that apply)']).split(';')
                for domain in user_domains:
                    domain_name = domain.split('(')[0].strip()
                    if domain_name.lower() in course['skills'].lower():
                        domain_match = 1.0
                        break
            
            # Create a synthetic satisfaction score (target)
            # Higher when matches are good, lower when matches are poor
            satisfaction = (0.4 * domain_match + 0.3 * level_match + 0.3 * cost_match) 
            satisfaction = np.clip(satisfaction + np.random.normal(0, 0.1), 0, 1)  # Add some noise
            
            # Add to training data
            training_data.append({
                'domain_match': domain_match,
                'level_match': level_match,
                'cost_match': cost_match,
                'course_rating': course.get('normalized_rating', 0.5),
                'satisfaction': 1 if satisfaction > 0.6 else 0  # Binary classification
            })
    
    # Convert to DataFrame
    train_df = pd.DataFrame(training_data)
    
    # Train Random Forest model
    X = train_df[['domain_match', 'level_match', 'cost_match', 'course_rating']]
    y = train_df['satisfaction']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

def recommend_courses_for_user(user_profile, courses_df, tfidf_vectorizer, tfidf_matrix, rf_model=None, top_n=5):
    """Recommend courses for a user based on their profile using hybrid approach"""
    # Extract user preferences
    domain_of_interest = user_profile.get('domain_of_interest', '')
    cost_preference = user_profile.get('cost_preference', 'Free')
    knowledge_level = user_profile.get('knowledge_level', 'Beginner')
    
    # Preprocess domain of interest
    domain_of_interest = preprocess_text(domain_of_interest)
    
    # Filter courses based on cost preference
    if cost_preference == 'Free':
        filtered_courses = courses_df[courses_df['cost_type'] == 'Free']
    elif cost_preference == 'Paid':
        filtered_courses = courses_df[courses_df['cost_type'] == 'Paid']
    else:
        filtered_courses = courses_df
    
    # Filter courses based on knowledge level
    if knowledge_level == 'Beginner':
        level_filter = filtered_courses['knowledge_level'].isin(['Beginner', 'All Levels'])
    elif knowledge_level == 'Intermediate':
        level_filter = filtered_courses['knowledge_level'].isin(['Intermediate', 'All Levels'])
    elif knowledge_level == 'Advanced':
        level_filter = filtered_courses['knowledge_level'].isin(['Advanced', 'All Levels'])
    else:
        level_filter = True
    
    filtered_courses = filtered_courses[level_filter]
    
    # If no courses match the filters, return empty DataFrame
    if filtered_courses.empty:
        return pd.DataFrame()
    
    # Vectorize user's domain of interest
    user_vector = tfidf_vectorizer.transform([domain_of_interest])
    
    # Calculate similarity scores between user's interest and filtered courses
    indices = filtered_courses.index.tolist()
    
    # Using linear_kernel for cosine similarity (faster than cosine_similarity for TF-IDF)
    similarity_scores = linear_kernel(user_vector, tfidf_matrix[indices]).flatten()
    
    # Create a DataFrame with filtered courses and their similarity scores
    recommendations = filtered_courses.copy()
    recommendations['similarity_score'] = similarity_scores
    
    # If we have a Random Forest model, use it to enhance recommendations
    if rf_model is not None:
        # Calculate features for Random Forest
        recommendations['domain_match'] = recommendations['similarity_score']
        
        # Level match
        recommendations['level_match'] = 0.0
        if knowledge_level == 'Beginner':
            recommendations.loc[recommendations['knowledge_level'].isin(['Beginner', 'All Levels']), 'level_match'] = 1.0
        elif knowledge_level == 'Intermediate':
            recommendations.loc[recommendations['knowledge_level'].isin(['Intermediate', 'All Levels']), 'level_match'] = 1.0
        elif knowledge_level == 'Advanced':
            recommendations.loc[recommendations['knowledge_level'].isin(['Advanced', 'All Levels']), 'level_match'] = 1.0
        
        # Cost match
        recommendations['cost_match'] = 0.0
        recommendations.loc[recommendations['cost_type'] == cost_preference, 'cost_match'] = 1.0
        
        # Predict satisfaction using Random Forest
        X_pred = recommendations[['domain_match', 'level_match', 'cost_match', 'normalized_rating']]
        recommendations['rf_score'] = rf_model.predict_proba(X_pred)[:, 1]  # Probability of class 1 (satisfaction)
        
        # Calculate final score combining similarity, rating, and RF prediction
        recommendations['final_score'] = (
            0.4 * recommendations['similarity_score'] + 
            0.3 * recommendations['normalized_rating'] +
            0.3 * recommendations['rf_score']
        )
    else:
        # Calculate final score combining similarity and rating
        recommendations['final_score'] = (
            SIMILARITY_WEIGHT * recommendations['similarity_score'] + 
            RATING_WEIGHT * recommendations['normalized_rating']
        )
    
    # Sort by final score and return top N recommendations
    recommendations = recommendations.sort_values('final_score', ascending=False).head(top_n)
    
    return recommendations[['course_id', 'course_title', 'level', 'cost_type', 'price', 
                          'similarity_score', 'final_score', 'url']]

def parse_user_survey_data(survey_row):
    """Parse user data from survey response"""
    user_profile = {
        'domain_of_interest': '',
        'cost_preference': 'Free',
        'knowledge_level': 'Beginner'
    }
    
    # Extract domain of interest
    domain_field = 'Domain of Interest: (Select all that apply)'
    if domain_field in survey_row and pd.notna(survey_row[domain_field]):
        domain_interests = []
        interests = str(survey_row[domain_field]).split(';')
        for interest in interests:
            main_category = interest.split('(')[0].strip()
            domain_interests.append(main_category)
        
        user_profile['domain_of_interest'] = ' '.join(domain_interests)
    
    # Extract cost preference
    cost_field = 'Course Cost: (Select your preference)'
    if cost_field in survey_row and pd.notna(survey_row[cost_field]):
        if 'Free' in str(survey_row[cost_field]):
            user_profile['cost_preference'] = 'Free'
        elif 'Paid' in str(survey_row[cost_field]):
            user_profile['cost_preference'] = 'Paid'
    
    # Extract knowledge level
    knowledge_field = 'Your Knowledge Level: (Select one)'
    if knowledge_field in survey_row and pd.notna(survey_row[knowledge_field]):
        if 'Beginner' in str(survey_row[knowledge_field]):
            user_profile['knowledge_level'] = 'Beginner'
        elif 'Intermediate' in str(survey_row[knowledge_field]):
            user_profile['knowledge_level'] = 'Intermediate'
        elif 'Advanced' in str(survey_row[knowledge_field]):
            user_profile['knowledge_level'] = 'Advanced'
    
    return user_profile

def extract_knowledge_domains(survey_df):
    """Extract unique knowledge domains from survey data"""
    domains = set()
    domain_field = 'Domain of Interest: (Select all that apply)'
    
    if domain_field in survey_df.columns:
        for interests in survey_df[domain_field].dropna():
            for interest in str(interests).split(';'):
                domain = interest.split('(')[0].strip()
                if domain:
                    domains.add(domain)
    
    return sorted(list(domains))

class CourseRecommender:
    """A class to handle course recommendations"""
    def __init__(self, course_data_path='coursera_1000.csv', user_data_path='Course-Recommender-System.csv'):
        """Initialize the recommender"""
        self.courses_df = load_course_data(course_data_path)
        self.tfidf_vectorizer, self.tfidf_matrix = create_tfidf_vectorizer(self.courses_df)
        
        # Load user data if available
        self.user_df = None
        self.rf_model = None
        
        if os.path.exists(user_data_path):
            try:
                self.user_df = pd.read_csv(user_data_path)
                # Train Random Forest model
                self.rf_model = train_random_forest_model(self.user_df, self.courses_df)
            except Exception as e:
                st.warning(f"Could not load user data or train model: {str(e)}")
    
    def recommend_courses(self, user_profile, top_n=5):
        """Recommend courses for a user"""
        return recommend_courses_for_user(
            user_profile, 
            self.courses_df, 
            self.tfidf_vectorizer, 
            self.tfidf_matrix,
            self.rf_model,
            top_n
        )
    
    def recommend_courses_from_survey(self, survey_row, top_n=5):
        """Recommend courses based on survey response"""
        user_profile = parse_user_survey_data(survey_row)
        return self.recommend_courses(user_profile, top_n)

def run_course_recommendation(reset_callback):
    """Run the course recommendation section of the app"""
    st.markdown("<h1 class='sub-header'>📚 Course Recommendation Engine</h1>", unsafe_allow_html=True)
    st.button("← Back to Home", on_click=reset_callback)
    
    st.markdown("""
        This recommendation engine suggests courses based on your interests and preferences.
        Fill out the form below to get personalized course recommendations.
    """)
    
    try:
        # File uploader for course data
        st.sidebar.header("Upload Data Files")
        course_file = st.sidebar.file_uploader(
            "Upload course data (CSV)",
            type=["csv"],
            key="course_data"
        )
        
        # File uploader for user survey data
        survey_file = st.sidebar.file_uploader(
            "Upload user survey data (CSV)",
            type=["csv"],
            key="survey_data"
        )
        
        # Initialize the recommender with courses
        if course_file is not None:
            course_data_path = course_file
            st.sidebar.success("✅ Course data loaded successfully!")
        else:
            course_data_path = 'coursera_1000.csv'
            if not os.path.exists(course_data_path):
                st.sidebar.warning(f"Default course data file '{course_data_path}' not found. Please upload a file.")
        
        if survey_file is not None:
            user_data_path = survey_file
            st.sidebar.success("✅ User survey data loaded successfully!")
        else:
            user_data_path = 'Course-Recommender-System.csv'
            if not os.path.exists(user_data_path):
                st.sidebar.info(f"Default user data file '{user_data_path}' not found. Recommendations will use content-based filtering only.")
        
        recommender = CourseRecommender(course_data_path, user_data_path)
        
        # Dashboard layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("<h3>🧑‍💻 Your Preferences</h3>", unsafe_allow_html=True)
            
            # User profile input form
            with st.form("user_profile_form"):
                # Domains of interest
                if recommender.user_df is not None:
                    domains = extract_knowledge_domains(recommender.user_df)
                    domain_options = domains if domains else ["Programming", "Data Science", "Business", "Design", "Marketing"]
                else:
                    domain_options = ["Programming", "Data Science", "Business", "Design", "Marketing"]
                
                selected_domains = st.multiselect(
                    "Select your domains of interest:",
                    options=domain_options,
                    default=domain_options[:1] if domain_options else []
                )
                
                # Cost preference
                cost_preference = st.selectbox(
                    "Course cost preference:",
                    options=["Any", "Free", "Paid"],
                    index=0
                )
                
                # Knowledge level
                knowledge_level = st.selectbox(
                    "Your knowledge level:",
                    options=["Beginner", "Intermediate", "Advanced", "All Levels"],
                    index=0
                )
                
                # Number of recommendations
                num_recommendations = st.slider(
                    "Number of recommendations:",
                    min_value=3,
                    max_value=10,
                    value=5
                )
                
                # Additional preferences (optional)
                with st.expander("Additional Preferences (Optional)"):
                    max_duration = st.slider(
                        "Maximum course duration (hours):",
                        min_value=1,
                        max_value=20,
                        value=10
                    )
                    
                    min_rating = st.slider(
                        "Minimum course rating:",
                        min_value=1.0,
                        max_value=5.0,
                        value=4.0,
                        step=0.1
                    )
                
                # Submit button
                submitted = st.form_submit_button("Get Recommendations")
            
            # User profile from survey data
            if recommender.user_df is not None:
                st.markdown("<h3>📊 or Select from Survey</h3>", unsafe_allow_html=True)
                
                if not recommender.user_df.empty:
                    # Get unique usernames
                    if 'Username' in recommender.user_df.columns:
                        usernames = recommender.user_df['Username'].dropna().unique()
                        
                        if len(usernames) > 0:
                            selected_user = st.selectbox(
                                "Select a user from survey data:",
                                options=usernames
                            )
                            
                            use_survey_data = st.button("Use This User's Preferences")
                        else:
                            st.warning("No usernames found in survey data")
                            use_survey_data = False
                    else:
                        st.warning("Username column not found in survey data")
                        use_survey_data = False
                else:
                    st.warning("Survey data is empty")
                    use_survey_data = False
        
        with col2:
            st.markdown("<h3>🎓 Recommended Courses</h3>", unsafe_allow_html=True)
            
            # Process recommendations based on form submission
            if submitted:
                if not selected_domains:
                    st.error("Please select at least one domain of interest")
                else:
                    # Create user profile
                    user_profile = {
                        'domain_of_interest': ' '.join(selected_domains),
                        'cost_preference': cost_preference,
                        'knowledge_level': knowledge_level
                    }
                    
                    with st.spinner('Finding the best courses for you...'):
                        # Get recommendations
                        recommendations = recommender.recommend_courses(user_profile, num_recommendations)
                        
                        if recommendations.empty:
                            st.warning("No courses match your preferences. Try adjusting your filters.")
                        else:
                            # Display recommendations
                            for i, (_, course) in enumerate(recommendations.iterrows()):
                                with st.container():
                                    st.markdown(f"""
                                    <div class='feature-card recommendation-card'>
                                        <h4>{i+1}. {course['course_title']}</h4>
                                        <p><b>Level:</b> {course['level']} | <b>Cost:</b> {course['cost_type']} {f"(${course['price']:.2f})" if course['cost_type'] == 'Paid' and not pd.isna(course['price']) else ''}</p>
                                        <p><b>Match Score:</b> {course['similarity_score']*100:.1f}% | <b>Overall Score:</b> {course['final_score']*100:.1f}%</p>
                                        <p><a href="{course['url']}" target="_blank">View Course</a></p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    st.markdown("---")
            
            # Process recommendations based on survey selection
            elif 'use_survey_data' in locals() and use_survey_data and 'selected_user' in locals():
                with st.spinner('Finding the best courses based on survey data...'):
                    # Get the selected user's data
                    user_row = recommender.user_df[recommender.user_df['Username'] == selected_user].iloc[0]
                    
                    # Get recommendations from survey data
                    recommendations = recommender.recommend_courses_from_survey(user_row, num_recommendations)
                    
                    if recommendations.empty:
                        st.warning("No courses match this user's preferences. Try another user or adjust filters manually.")
                    else:
                        # Display user preferences from survey
                        user_profile = parse_user_survey_data(user_row)
                        
                        st.markdown("<h4>User's Preferences:</h4>", unsafe_allow_html=True)
                        st.markdown(f"**Domains of Interest:** {user_profile['domain_of_interest']}")
                        st.markdown(f"**Cost Preference:** {user_profile['cost_preference']}")
                        st.markdown(f"**Knowledge Level:** {user_profile['knowledge_level']}")
                        st.markdown("---")
                        
                        # Display recommendations
                        for i, (_, course) in enumerate(recommendations.iterrows()):
                            with st.container():
                                st.markdown(f"""
                                <div class='feature-card recommendation-card'>
                                    <h4>{i+1}. {course['course_title']}</h4>
                                    <p><b>Level:</b> {course['level']} | <b>Cost:</b> {course['cost_type']} {f"(${course['price']:.2f})" if course['cost_type'] == 'Paid' and not pd.isna(course['price']) else ''}</p>
                                    <p><b>Match Score:</b> {course['similarity_score']*100:.1f}% | <b>Overall Score:</b> {course['final_score']*100:.1f}%</p>
                                    <p><a href="{course['url']}" target="_blank">View Course</a></p>
                                </div>
                                """, unsafe_allow_html=True)
                                st.markdown("---")
            else:
                # Display placeholder when no recommendations yet
                st.info("Fill out the form and click 'Get Recommendations' to see personalized course suggestions.")
                
                # Alternative courses display for exploration
                with st.expander("Browse Popular Courses", expanded=False):
                    # Simply show top-rated courses as a fallback
                    popular_courses = recommender.courses_df.sort_values('normalized_rating', ascending=False).head(5)
                    for i, (_, course) in enumerate(popular_courses.iterrows()):
                        st.markdown(f"**{i+1}. {course['course_title']}**")
                        st.markdown(f"Level: {course['level']} | Cost: {course['cost_type']}")
                        if 'url' in course and pd.notna(course['url']):
                            st.markdown(f"[View Course]({course['url']})")
                        st.markdown("---")
        
        # Analytics section
        with st.expander("📊 Recommendation Analytics", expanded=False):
            st.markdown("### Insights from Course Data")
            
            # Calculate analytics
            courses_df = recommender.courses_df
            
            # Distribution of courses by level
            level_counts = courses_df['knowledge_level'].value_counts().reset_index()
            level_counts.columns = ['Level', 'Count']
            
            # Distribution of courses by cost type
            cost_counts = courses_df['cost_type'].value_counts().reset_index()
            cost_counts.columns = ['Cost Type', 'Count']
            
            # Layout for charts
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.subheader("Courses by Level")
                st.bar_chart(level_counts.set_index('Level'))
            
            with chart_col2:
                st.subheader("Courses by Cost")
                st.bar_chart(cost_counts.set_index('Cost Type'))
            
            # Course subjects analysis
            if 'subject' in courses_df.columns:
                subject_counts = courses_df['subject'].value_counts().head(10).reset_index()
                subject_counts.columns = ['Subject', 'Count']
                
                st.subheader("Top Course Subjects")
                st.bar_chart(subject_counts.set_index('Subject'))
            
            # Model information
            if recommender.rf_model is not None:
                st.subheader("Random Forest Model Information")
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'Feature': ['Domain Match', 'Level Match', 'Cost Match', 'Course Rating'],
                    'Importance': recommender.rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.write("Feature Importance:")
                st.bar_chart(feature_importance.set_index('Feature'))
                
                st.write(f"Number of trees in the forest: {recommender.rf_model.n_estimators}")
                st.write(f"Model accuracy on training data: {recommender.rf_model.score(feature_importance['Importance'].values.reshape(-1, 1), feature_importance['Importance'] > 0.2):.2f}")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

if __name__ == "__main__":
    # This allows running this file directly for testing
    st.set_page_config(page_title="Course Recommender", layout="wide")
    
    # Placeholder for reset callback
    def temp_reset():
        pass
    
    # Run the recommendation page
    run_course_recommendation(temp_reset)
