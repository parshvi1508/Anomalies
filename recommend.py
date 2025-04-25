# course_recommendations.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import streamlit as st
import re
import os
from scipy.stats import randint
import tempfile

# Constants
SIMILARITY_WEIGHT = 0.4
RATING_WEIGHT = 0.3
RF_WEIGHT = 0.3

def save_uploaded_file(uploaded_file):
    """Save an uploaded file to a temporary directory and return the path"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
        # Write bytes directly from the uploaded file buffer
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name

def load_course_data(file):
    """Load and preprocess course data"""
    try:
        if isinstance(file, str):
            courses_df = pd.read_csv(file)
        else:
            # Handle uploaded file
            file_path = save_uploaded_file(file)
            courses_df = pd.read_csv(file_path)
            # Clean up temporary file
            os.unlink(file_path)
        
        courses_df = courses_df.fillna('')
        
        # Create a skills field from course_title and subject if available
        if 'course_title' in courses_df.columns and 'subject' in courses_df.columns:
            courses_df['skills'] = courses_df['course_title'] + ' ' + courses_df['subject']
        elif 'course_title' in courses_df.columns:
            courses_df['skills'] = courses_df['course_title']
        else:
            st.warning("Course title column not found. Recommendation quality may be reduced.")
            courses_df['skills'] = ''
        
        # Ensure needed columns exist
        if 'price' not in courses_df.columns:
            courses_df['price'] = 0.0
        else:
            courses_df['price'] = pd.to_numeric(courses_df['price'], errors='coerce')
        
        # Calculate normalized rating
        calculate_normalized_rating(courses_df)
        
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
        
        # Ensure course_id exists
        if 'course_id' not in courses_df.columns:
            courses_df['course_id'] = courses_df.index
        
        return courses_df
    except Exception as e:
        st.error(f"Error loading course data: {str(e)}")
        return pd.DataFrame()

def calculate_normalized_rating(courses_df):
    """Calculate normalized rating from available metrics"""
    if 'num_reviews' in courses_df.columns and 'num_subscribers' in courses_df.columns:
        courses_df['normalized_reviews'] = normalize_column(courses_df, 'num_reviews')
        courses_df['normalized_subscribers'] = normalize_column(courses_df, 'num_subscribers')
        
        # Combine normalized reviews and subscribers as a proxy for rating
        courses_df['normalized_rating'] = (
            0.7 * courses_df['normalized_reviews'] + 
            0.3 * courses_df['normalized_subscribers']
        )
    elif 'rating' in courses_df.columns:
        # If direct rating is available
        courses_df['normalized_rating'] = normalize_column(courses_df, 'rating')
    else:
        courses_df['normalized_rating'] = 0.5  # Default if no ratings

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

def train_random_forest_model(courses_df, user_df=None):
    """Train a Random Forest model for enhanced recommendations"""
    # If we don't have user data, create synthetic data based on course features
    if user_df is None or user_df.empty:
        st.info("Training Random Forest on course features only (no user data available)")
        # Extract features from courses for training
        features = []
        labels = []
        
        # Convert categorical features to numeric
        level_encoder = LabelEncoder()
        courses_df['level_encoded'] = level_encoder.fit_transform(courses_df['knowledge_level'])
        
        cost_encoder = LabelEncoder()
        courses_df['cost_encoded'] = cost_encoder.fit_transform(courses_df['cost_type'])
        
        # Create synthetic training data based on course features
        for idx, course in courses_df.iterrows():
            # Features: level, cost, normalized_rating
            features.append([
                course['level_encoded'],
                course['cost_encoded'],
                course['normalized_rating']
            ])
            
            # Label: high rating (1) or low rating (0)
            # Assuming courses with normalized_rating > 0.7 are "good courses"
            labels.append(1 if course['normalized_rating'] > 0.7 else 0)
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
    else:
        st.info("Training Random Forest with user data")
        # Extract user-course interactions
        # This is a simplified example - in a real system, you'd use actual user ratings
        
        # Create features based on user-course matches
        features = []
        labels = []
        
        # Process each user to extract training data
        for _, user in user_df.iterrows():
            user_profile = parse_user_survey_data(user)
            domain_of_interest = user_profile.get('domain_of_interest', '')
            cost_preference = user_profile.get('cost_preference', 'Any')
            knowledge_level = user_profile.get('knowledge_level', 'All Levels')
            
            for idx, course in courses_df.iterrows():
                # Domain match feature
                domain_match = 1.0 if domain_of_interest.lower() in course['skills'].lower() else 0.0
                
                # Cost match feature
                cost_match = 1.0
                if cost_preference != 'Any':
                    cost_match = 1.0 if course['cost_type'] == cost_preference else 0.0
                
                # Level match feature
                level_match = 1.0
                if knowledge_level != 'All Levels':
                    level_match = 1.0 if course['knowledge_level'] == knowledge_level or course['knowledge_level'] == 'All Levels' else 0.0
                
                # Features: domain match, cost match, level match, normalized rating
                features.append([domain_match, cost_match, level_match, course['normalized_rating']])
                
                # Synthetic label: good match (1) or poor match (0)
                match_score = (0.5 * domain_match + 0.3 * cost_match + 0.2 * level_match)
                labels.append(1 if match_score > 0.7 else 0)
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
    
    # Train Random Forest with hyperparameter tuning
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(5, 20),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5)
    }
    
    # Create a pipeline with scaling and random forest
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42))
    ])
    
    # Use RandomizedSearchCV for hyperparameter tuning
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions={'rf__' + key: val for key, val in param_dist.items()},
        n_iter=10,
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    
    try:
        random_search.fit(X, y)
        st.success(f"Random Forest model trained successfully with best parameters: {random_search.best_params_}")
        return random_search.best_estimator_
    except Exception as e:
        st.warning(f"Error training Random Forest model: {str(e)}. Using default model.")
        # Fall back to default model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        pipeline.fit(X, y)
        return pipeline

def collaborative_filtering_recommendations(user_profile, courses_df, user_df=None, top_n=10):
    """Generate recommendations using collaborative filtering"""
    if user_df is None or user_df.empty:
        # If no user data is available, return empty DataFrame
        return pd.DataFrame()
    
    # Parse user profile
    domain_of_interest = user_profile.get('domain_of_interest', '')
    cost_preference = user_profile.get('cost_preference', 'Any')
    knowledge_level = user_profile.get('knowledge_level', 'All Levels')
    
    # Find similar users based on preferences
    similar_users = []
    
    for _, user in user_df.iterrows():
        other_user_profile = parse_user_survey_data(user)
        other_domain = other_user_profile.get('domain_of_interest', '')
        other_cost = other_user_profile.get('cost_preference', 'Any')
        other_level = other_user_profile.get('knowledge_level', 'All Levels')
        
        # Calculate similarity score between users
        domain_similarity = 0
        if domain_of_interest and other_domain:
            # Simple text overlap for domain similarity
            domain_similarity = len(set(domain_of_interest.lower().split()) & set(other_domain.lower().split())) / \
                               max(len(set(domain_of_interest.lower().split())), 1)
        
        cost_similarity = 1 if cost_preference == other_cost else 0
        level_similarity = 1 if knowledge_level == other_level else 0
        
        # Weighted similarity score
        similarity_score = (0.6 * domain_similarity + 0.2 * cost_similarity + 0.2 * level_similarity)
        
        similar_users.append((user, similarity_score))
    
    # Sort by similarity score (descending)
    similar_users.sort(key=lambda x: x[1], reverse=True)
    
    # Take top similar users
    top_similar_users = similar_users[:min(5, len(similar_users))]
    
    # Collect course preferences from similar users
    course_scores = {}
    
    for user, similarity in top_similar_users:
        if similarity <= 0:
            continue
            
        # Extract course preferences from user profile
        user_profile = parse_user_survey_data(user)
        domain_of_interest = user_profile.get('domain_of_interest', '')
        
        # Find matching courses based on domain
        for idx, course in courses_df.iterrows():
            # Simple matching based on domain overlap
            if domain_of_interest.lower() in course['skills'].lower():
                course_id = course['course_id']
                if course_id not in course_scores:
                    course_scores[course_id] = 0
                course_scores[course_id] += similarity
    
    # Sort courses by score
    ranked_courses = sorted(course_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get top N courses
    top_course_ids = [course_id for course_id, _ in ranked_courses[:top_n]]
    
    # Filter courses DataFrame to include only recommended courses
    recommended_courses = courses_df[courses_df['course_id'].isin(top_course_ids)].copy()
    
    # Add collaborative filtering score
    recommended_courses['cf_score'] = recommended_courses['course_id'].apply(
        lambda x: course_scores.get(x, 0)
    )
    
    # Normalize the CF score
    if not recommended_courses.empty and recommended_courses['cf_score'].max() > 0:
        recommended_courses['cf_score'] = recommended_courses['cf_score'] / recommended_courses['cf_score'].max()
    
    return recommended_courses

def content_based_recommendations(user_profile, courses_df, tfidf_vectorizer, tfidf_matrix, top_n=10):
    """Generate recommendations using content-based filtering"""
    # Extract user preferences
    domain_of_interest = user_profile.get('domain_of_interest', '')
    cost_preference = user_profile.get('cost_preference', 'Any')
    knowledge_level = user_profile.get('knowledge_level', 'All Levels')
    
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
    recommendations['cb_score'] = similarity_scores
    
    # Sort by similarity score and return top N recommendations
    recommendations = recommendations.sort_values('cb_score', ascending=False).head(top_n)
    
    return recommendations

def random_forest_recommendations(user_profile, courses_df, rf_model, top_n=10):
    """Generate recommendations using Random Forest model"""
    # Extract user preferences
    domain_of_interest = user_profile.get('domain_of_interest', '')
    cost_preference = user_profile.get('cost_preference', 'Any')
    knowledge_level = user_profile.get('knowledge_level', 'All Levels')
    
    # Prepare features for each course
    features = []
    course_indices = []
    
    for idx, course in courses_df.iterrows():
        # Domain match feature (simple text matching)
        domain_match = 1.0 if domain_of_interest.lower() in course['skills'].lower() else 0.0
        
        # Cost match feature
        cost_match = 1.0
        if cost_preference != 'Any':
            cost_match = 1.0 if course['cost_type'] == cost_preference else 0.0
        
        # Level match feature
        level_match = 1.0
        if knowledge_level != 'All Levels':
            level_match = 1.0 if course['knowledge_level'] == knowledge_level or course['knowledge_level'] == 'All Levels' else 0.0
        
        # Features: domain match, cost match, level match, normalized rating
        features.append([domain_match, cost_match, level_match, course['normalized_rating']])
        course_indices.append(idx)
    
    # If no courses available, return empty DataFrame
    if not features:
        return pd.DataFrame()
    
    # Convert to numpy array
    X = np.array(features)
    
    # Predict probability of class 1 (good match)
    try:
        # Use the model to predict probabilities
        proba = rf_model.predict_proba(X)[:, 1]
        
        # Create a DataFrame with courses and their RF scores
        recommendations = courses_df.iloc[course_indices].copy()
        recommendations['rf_score'] = proba
        
        # Sort by RF score and return top N recommendations
        recommendations = recommendations.sort_values('rf_score', ascending=False).head(top_n)
        
        return recommendations
    except Exception as e:
        st.warning(f"Error making Random Forest predictions: {str(e)}")
        return pd.DataFrame()

def hybrid_recommendations(user_profile, courses_df, tfidf_vectorizer, tfidf_matrix, rf_model, user_df=None, top_n=5):
    """Generate recommendations using hybrid approach"""
    # Get recommendations from content-based filtering
    cb_recommendations = content_based_recommendations(
        user_profile, courses_df, tfidf_vectorizer, tfidf_matrix, top_n=top_n*2
    )
    
    # Get recommendations from Random Forest
    rf_recommendations = random_forest_recommendations(
        user_profile, courses_df, rf_model, top_n=top_n*2
    )
    
    # Get recommendations from collaborative filtering
    cf_recommendations = collaborative_filtering_recommendations(
        user_profile, courses_df, user_df, top_n=top_n*2
    )
    
    # Combine all recommendations
    all_course_ids = set()
    if not cb_recommendations.empty:
        all_course_ids.update(cb_recommendations['course_id'])
    if not rf_recommendations.empty:
        all_course_ids.update(rf_recommendations['course_id'])
    if not cf_recommendations.empty:
        all_course_ids.update(cf_recommendations['course_id'])
    
    # If no recommendations, return empty DataFrame
    if not all_course_ids:
        return pd.DataFrame()
    
    # Create a DataFrame with all recommended courses
    hybrid_df = courses_df[courses_df['course_id'].isin(all_course_ids)].copy()
    
    # Add scores from each approach (default to 0 if not present)
    hybrid_df['cb_score'] = 0.0
    if not cb_recommendations.empty:
        for idx, row in cb_recommendations.iterrows():
            course_id = row['course_id']
            mask = hybrid_df['course_id'] == course_id
            if any(mask):
                hybrid_df.loc[mask, 'cb_score'] = row['cb_score']
    
    hybrid_df['rf_score'] = 0.0
    if not rf_recommendations.empty:
        for idx, row in rf_recommendations.iterrows():
            course_id = row['course_id']
            mask = hybrid_df['course_id'] == course_id
            if any(mask):
                hybrid_df.loc[mask, 'rf_score'] = row['rf_score']
    
    hybrid_df['cf_score'] = 0.0
    if not cf_recommendations.empty:
        for idx, row in cf_recommendations.iterrows():
            course_id = row['course_id']
            mask = hybrid_df['course_id'] == course_id
            if any(mask):
                hybrid_df.loc[mask, 'cf_score'] = row['cf_score']
    
    # Calculate final score using weighted approach
    if not cf_recommendations.empty:
        # If we have collaborative filtering data, use all three approaches
        hybrid_df['final_score'] = (
            SIMILARITY_WEIGHT * hybrid_df['cb_score'] +
            RF_WEIGHT * hybrid_df['rf_score'] +
            RATING_WEIGHT * hybrid_df['cf_score']
        )
    else:
        # If no collaborative filtering data, adjust weights for content-based and RF
        hybrid_df['final_score'] = (
            0.6 * hybrid_df['cb_score'] +
            0.4 * hybrid_df['rf_score']
        )
    
    # Sort by final score and return top N recommendations
    hybrid_df = hybrid_df.sort_values('final_score', ascending=False).head(top_n)
    
    return hybrid_df[['course_id', 'course_title', 'level', 'cost_type', 'price', 
                      'cb_score', 'rf_score', 'cf_score', 'final_score', 'url']]

def parse_user_survey_data(survey_row):
    """Parse user data from survey response"""
    user_profile = {
        'domain_of_interest': '',
        'cost_preference': 'Any',
        'knowledge_level': 'All Levels'
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

class HybridCourseRecommender:
    """A class to handle hybrid course recommendations"""
    def __init__(self, course_data=None, user_data=None):
        """Initialize the recommender with optional course and user data"""
        self.courses_df = None
        self.user_df = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.rf_model = None
        
        # Load data if provided
        if course_data is not None:
            self.load_course_data(course_data)
        
        if user_data is not None:
            self.load_user_data(user_data)
    
    def load_course_data(self, course_data):
        """Load course data from file or DataFrame"""
        if isinstance(course_data, pd.DataFrame):
            self.courses_df = course_data
        else:
            self.courses_df = load_course_data(course_data)
        
        # Create TF-IDF vectorizer
        if not self.courses_df.empty:
            self.tfidf_vectorizer, self.tfidf_matrix = create_tfidf_vectorizer(self.courses_df)
            
            # Train RF model if user data is available
            if self.user_df is not None:
                self.rf_model = train_random_forest_model(self.courses_df, self.user_df)
            else:
                self.rf_model = train_random_forest_model(self.courses_df)
    
    def load_user_data(self, user_data):
        """Load user data from file or DataFrame"""
        if isinstance(user_data, pd.DataFrame):
            self.user_df = user_data
        else:
            try:
                if isinstance(user_data, str):
                    self.user_df = pd.read_csv(user_data)
                    st.sidebar.success(f"Successfully loaded user data from {user_data}")
                else:
                    # Handle uploaded file
                    file_path = save_uploaded_file(user_data)
                    self.user_df = pd.read_csv(file_path)
                    st.sidebar.success(f"Successfully loaded uploaded user data (size: {user_data.size} bytes)")
                    # Clean up temporary file
                    os.unlink(file_path)
            except Exception as e:
                st.error(f"Error loading user data: {str(e)}")
                self.user_df = pd.DataFrame()
        
        # Train RF model if course data is available
        if self.courses_df is not None and not self.courses_df.empty:
            self.rf_model = train_random_forest_model(self.courses_df, self.user_df)
    
    def recommend_courses(self, user_profile, top_n=5, recommendation_type='hybrid'):
        """Recommend courses for a user"""
        if self.courses_df is None or self.courses_df.empty:
            return pd.DataFrame()
        
        if recommendation_type == 'content':
            return content_based_recommendations(
                user_profile, 
                self.courses_df, 
                self.tfidf_vectorizer, 
                self.tfidf_matrix, 
                top_n
            )
        elif recommendation_type == 'collaborative':
            return collaborative_filtering_recommendations(
                user_profile,
                self.courses_df,
                self.user_df,
                top_n
            )
        elif recommendation_type == 'random_forest':
            if self.rf_model is None:
                self.rf_model = train_random_forest_model(self.courses_df, self.user_df)
            
            return random_forest_recommendations(
                user_profile,
                self.courses_df,
                self.rf_model,
                top_n
            )
        else:  # hybrid (default)
            if self.rf_model is None:
                self.rf_model = train_random_forest_model(self.courses_df, self.user_df)
            
            return hybrid_recommendations(
                user_profile,
                self.courses_df,
                self.tfidf_vectorizer,
                self.tfidf_matrix,
                self.rf_model,
                self.user_df,
                top_n
            )
    
    def recommend_courses_from_survey(self, survey_row, top_n=5, recommendation_type='hybrid'):
        """Recommend courses based on survey response"""
        user_profile = parse_user_survey_data(survey_row)
        return self.recommend_courses(user_profile, top_n, recommendation_type)

def run_course_recommendation(reset_callback):
    """Run the course recommendation section of the app"""
    st.markdown("<h1 class='sub-header'>📚 Course Recommendation Engine</h1>", unsafe_allow_html=True)
    st.button("← Back to Home", on_click=reset_callback)
    
    st.markdown("""
        This advanced recommendation engine suggests courses based on your interests and preferences
        using a hybrid approach combining content-based filtering, collaborative filtering, and machine learning.
        
        Upload your own datasets or use the defaults to get personalized course recommendations.
    """)
    
    try:
        # File uploaders in sidebar
        st.sidebar.header("Upload Data Files")
        
        course_file = st.sidebar.file_uploader(
            "Upload course data (CSV)",
            type=["csv"],
            key="course_data"
        )
        
        user_file = st.sidebar.file_uploader(
            "Upload user survey data (CSV)",
            type=["csv"],
            key="user_data"
        )
        
        # Debug information for file uploads
        if course_file is not None:
            st.sidebar.info(f"Course file uploaded: {course_file.name}, size: {course_file.size} bytes")
            
        if user_file is not None:
            st.sidebar.info(f"User file uploaded: {user_file.name}, size: {user_file.size} bytes")
        
        # Initialize the recommender
        recommender = HybridCourseRecommender()
        
        # Load course data
        if course_file is not None:
            with st.spinner('Loading course data...'):
                recommender.load_course_data(course_file)
                st.sidebar.success("✅ Course data loaded successfully!")
        else:
            # Try to load default course data
            default_course_path = 'coursera_1000.csv'
            if os.path.exists(default_course_path):
                with st.spinner('Loading default course data...'):
                    recommender.load_course_data(default_course_path)
                    st.sidebar.info(f"Using default course data: {default_course_path}")
            else:
                st.sidebar.warning("No course data loaded. Please upload a CSV file.")
        
        # Load user data
        if user_file is not None:
            with st.spinner('Loading user survey data...'):
                recommender.load_user_data(user_file)
                st.sidebar.success("✅ User survey data loaded successfully!")
        else:
            # Try to load default user data
            default_user_path = 'Course-Recommender-System.csv'
            if os.path.exists(default_user_path):
                with st.spinner('Loading default user survey data...'):
                    recommender.load_user_data(default_user_path)
                    st.sidebar.info(f"Using default user data: {default_user_path}")
            else:
                st.sidebar.info("No user survey data loaded. Recommendations will use content-based filtering only.")
        
        # Recommendation type selection
        recommendation_type = st.sidebar.radio(
            "Recommendation Approach:",
            options=['Hybrid (Recommended)', 'Content-Based Only', 'Collaborative Only', 'Random Forest Only'],
            index=0
        )
        
        # Map selection to internal types
        recommendation_type_map = {
            'Hybrid (Recommended)': 'hybrid',
            'Content-Based Only': 'content',
            'Collaborative Only': 'collaborative',
            'Random Forest Only': 'random_forest'
        }
        
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
                
                # Advanced options
                with st.expander("Advanced Options", expanded=False):
                    # Recommendation weights
                    st.markdown("#### Recommendation Weights")
                    st.info("These weights determine how much each factor influences the final recommendations.")
                    
                    content_weight = st.slider(
                        "Content-based weight:",
                        min_value=0.0,
                        max_value=1.0,
                        value=SIMILARITY_WEIGHT,
                        step=0.1,
                        help="Weight for content similarity (how well the course matches your interests)"
                    )
                    
                    rating_weight = st.slider(
                        "Rating weight:",
                        min_value=0.0,
                        max_value=1.0,
                        value=RATING_WEIGHT,
                        step=0.1,
                        help="Weight for course ratings"
                    )
                    
                    rf_weight = st.slider(
                        "Machine learning weight:",
                        min_value=0.0,
                        max_value=1.0,
                        value=RF_WEIGHT,
                        step=0.1,
                        help="Weight for machine learning predictions"
                    )
                    
                    # Normalize weights
                    total = content_weight + rating_weight + rf_weight
                    if total > 0:
                        content_weight = content_weight / total
                        rating_weight = rating_weight / total
                        rf_weight = rf_weight / total
                
                # Submit button
                submitted = st.form_submit_button("Get Recommendations")
            
            # User profile from survey data
            if recommender.user_df is not None and not recommender.user_df.empty:
                st.markdown("<h3>📊 or Select from User Survey</h3>", unsafe_allow_html=True)
                
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
        
        with col2:
            st.markdown("<h3>🎓 Recommended Courses</h3>", unsafe_allow_html=True)
            
            # Check if we have course data
            if recommender.courses_df is None or recommender.courses_df.empty:
                st.error("No course data available. Please upload a course dataset to get recommendations.")
            else:
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
                            recommendations = recommender.recommend_courses(
                                user_profile,
                                num_recommendations,
                                recommendation_type_map[recommendation_type]
                            )
                            
                            if recommendations.empty:
                                st.warning("No courses match your preferences. Try adjusting your filters.")
                            else:
                                # Display recommendations
                                for i, (_, course) in enumerate(recommendations.iterrows()):
                                    with st.container():
                                        # Create explanation string based on recommendation type
                                        explanation = ""
                                        if recommendation_type == 'Hybrid (Recommended)':
                                            explanation = f"Content Match: {course.get('cb_score', 0)*100:.1f}% | ML Score: {course.get('rf_score', 0)*100:.1f}%"
                                            if 'cf_score' in course and course['cf_score'] > 0:
                                                explanation += f" | Similar Users: {course['cf_score']*100:.1f}%"
                                        elif recommendation_type == 'Content-Based Only':
                                            explanation = f"Content Match: {course.get('cb_score', 0)*100:.1f}%"
                                        elif recommendation_type == 'Collaborative Only':
                                            explanation = f"Similar Users Score: {course.get('cf_score', 0)*100:.1f}%"
                                        elif recommendation_type == 'Random Forest Only':
                                            explanation = f"Machine Learning Score: {course.get('rf_score', 0)*100:.1f}%"
                                        
                                        st.markdown(f"""
                                        <div class='feature-card recommendation-card'>
                                            <h4>{i+1}. {course['course_title']}</h4>
                                            <p><b>Level:</b> {course['level']} | <b>Cost:</b> {course['cost_type']} {f"(${course['price']:.2f})" if course['cost_type'] == 'Paid' and not pd.isna(course['price']) else ''}</p>
                                            <p><b>{explanation}</b></p>
                                            <p><b>Overall Score:</b> {course['final_score']*100:.1f}%</p>
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
                        recommendations = recommender.recommend_courses_from_survey(
                            user_row,
                            num_recommendations,
                            recommendation_type_map[recommendation_type]
                        )
                        
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
                                    # Create explanation string based on recommendation type
                                    explanation = ""
                                    if recommendation_type == 'Hybrid (Recommended)':
                                        explanation = f"Content Match: {course.get('cb_score', 0)*100:.1f}% | ML Score: {course.get('rf_score', 0)*100:.1f}%"
                                        if 'cf_score' in course and course['cf_score'] > 0:
                                            explanation += f" | Similar Users: {course['cf_score']*100:.1f}%"
                                    elif recommendation_type == 'Content-Based Only':
                                        explanation = f"Content Match: {course.get('cb_score', 0)*100:.1f}%"
                                    elif recommendation_type == 'Collaborative Only':
                                        explanation = f"Similar Users Score: {course.get('cf_score', 0)*100:.1f}%"
                                    elif recommendation_type == 'Random Forest Only':
                                        explanation = f"Machine Learning Score: {course.get('rf_score', 0)*100:.1f}%"
                                    
                                    st.markdown(f"""
                                    <div class='feature-card recommendation-card'>
                                        <h4>{i+1}. {course['course_title']}</h4>
                                        <p><b>Level:</b> {course['level']} | <b>Cost:</b> {course['cost_type']} {f"(${course['price']:.2f})" if course['cost_type'] == 'Paid' and not pd.isna(course['price']) else ''}</p>
                                        <p><b>{explanation}</b></p>
                                        <p><b>Overall Score:</b> {course['final_score']*100:.1f}%</p>
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
                        if recommender.courses_df is not None and not recommender.courses_df.empty:
                            popular_courses = recommender.courses_df.sort_values('normalized_rating', ascending=False).head(5)
                            for i, (_, course) in enumerate(popular_courses.iterrows()):
                                st.markdown(f"**{i+1}. {course['course_title']}**")
                                st.markdown(f"Level: {course['level']} | Cost: {course['cost_type']}")
                                if 'url' in course and pd.notna(course['url']):
                                    st.markdown(f"[View Course]({course['url']})")
                                st.markdown("---")
        
        # Model analytics section
        if recommender.courses_df is not None and not recommender.courses_df.empty:
            with st.expander("📊 Recommendation System Analytics", expanded=False):
                st.markdown("### Insights about the Recommendation Engine")
                
                # Model information if RF model is available
                if recommender.rf_model is not None:
                    st.subheader("Random Forest Model Information")
                    
                    # Get the RandomForestClassifier from the pipeline
                    rf = recommender.rf_model.named_steps['rf']
                    
                    # Display model parameters
                    st.markdown(f"**Number of trees:** {rf.n_estimators}")
                    st.markdown(f"**Max depth:** {rf.max_depth if rf.max_depth else 'None (unlimited)'}")
                    st.markdown(f"**Min samples split:** {rf.min_samples_split}")
                    st.markdown(f"**Min samples leaf:** {rf.min_samples_leaf}")
                    
                    # Feature importance
                    st.subheader("Feature Importance")
                    
                    # Create DataFrame for feature importance
                    feature_names = ['Domain Match', 'Cost Match', 'Level Match', 'Course Rating']
                    feature_importance = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': rf.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    # Display feature importance chart
                    st.bar_chart(feature_importance.set_index('Feature'))
                
                # Course dataset information
                st.subheader("Course Dataset Information")
                
                # Distribution of courses by level
                level_counts = recommender.courses_df['knowledge_level'].value_counts().reset_index()
                level_counts.columns = ['Level', 'Count']
                
                # Distribution of courses by cost type
                cost_counts = recommender.courses_df['cost_type'].value_counts().reset_index()
                cost_counts.columns = ['Cost Type', 'Count']
                
                # Layout for charts
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    st.markdown("**Courses by Level**")
                    st.bar_chart(level_counts.set_index('Level'))
                
                with chart_col2:
                    st.markdown("**Courses by Cost**")
                    st.bar_chart(cost_counts.set_index('Cost Type'))
                
                # Course subjects analysis
                if 'subject' in recommender.courses_df.columns:
                    subject_counts = recommender.courses_df['subject'].value_counts().head(10).reset_index()
                    subject_counts.columns = ['Subject', 'Count']
                    
                    st.markdown("**Top Course Subjects**")
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