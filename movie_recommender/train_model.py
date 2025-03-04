import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json
import ast
from pathlib import Path
import re

def clean_title(title):
    """Clean movie title to extract base name and number."""
    # Remove common suffixes and clean the title
    title = re.sub(r'\s*\([^)]*\)', '', title)  # Remove parenthetical content
    title = re.sub(r'\s*\[[^\]]*\]', '', title)  # Remove bracketed content
    
    # Extract number if present
    number_match = re.search(r'\d+$', title)
    number = number_match.group() if number_match else None
    
    # Remove number and clean up
    base_title = re.sub(r'\s*\d+$', '', title).strip()
    
    return base_title, number

def load_movie_data():
    """Load movie data from TMDB CSV files."""
    try:
        # Load movies data
        movies_df = pd.read_csv('../data/tmdb_5000_movies.csv')
        
        # Load credits data
        credits_df = pd.read_csv('../data/tmdb_5000_credits.csv')
        
        # Merge movies and credits data
        movies_df = movies_df.merge(credits_df, on='title')
        
        # Convert string representations of lists/dicts to actual objects
        movies_df['genres'] = movies_df['genres'].apply(ast.literal_eval)
        movies_df['keywords'] = movies_df['keywords'].apply(ast.literal_eval)
        movies_df['cast'] = movies_df['cast'].apply(ast.literal_eval)
        movies_df['crew'] = movies_df['crew'].apply(ast.literal_eval)
        
        # Extract relevant information
        movies_df['genres'] = movies_df['genres'].apply(lambda x: ' '.join([i['name'] for i in x]))
        movies_df['keywords'] = movies_df['keywords'].apply(lambda x: ' '.join([i['name'] for i in x]))
        movies_df['cast'] = movies_df['cast'].apply(lambda x: ' '.join([i['name'] for i in x[:5]]))
        movies_df['crew'] = movies_df['crew'].apply(lambda x: ' '.join([i['name'] for i in x[:5]]))
        
        # Clean up the dataframe
        movies_df = movies_df[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 
                              'release_date', 'vote_average', 'vote_count', 'runtime', 'budget', 'revenue']]
        movies_df.columns = ['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew',
                           'release_date', 'vote_average', 'vote_count', 'runtime', 'budget', 'revenue']
        
        return movies_df
    except Exception as e:
        print(f"Error loading movie data: {e}")
        return None

def preprocess_features(df):
    """Preprocess features for better recommendations."""
    # Fill missing values
    df['overview'] = df['overview'].fillna('')
    df['genres'] = df['genres'].fillna('')
    df['keywords'] = df['keywords'].fillna('')
    df['cast'] = df['cast'].fillna('')
    df['crew'] = df['crew'].fillna('')
    
    # Combine text features
    df['combined_features'] = (
        df['overview'] + ' ' + 
        df['genres'] + ' ' + 
        df['cast'] + ' ' + 
        df['crew'] + ' ' + 
        df['keywords']
    )
    
    # Normalize numerical features
    numerical_features = ['vote_average', 'vote_count', 'runtime', 'budget', 'revenue']
    for feature in numerical_features:
        df[feature] = df[feature].fillna(0)  # Fill missing values with 0
        df[feature] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())
    
    return df

def calculate_similarity_matrix(movies_df, tfidf_matrix):
    """Calculate similarity matrix considering movie series and related movies."""
    # Calculate content similarity
    content_similarity = cosine_similarity(tfidf_matrix)
    
    # Calculate series similarity
    series_similarity = np.zeros((len(movies_df), len(movies_df)))
    for i in range(len(movies_df)):
        for j in range(len(movies_df)):
            if i != j:
                # Get genres for both movies
                genres_i = set(movies_df.iloc[i]['genres'].split())
                genres_j = set(movies_df.iloc[j]['genres'].split())
                
                # Calculate genre overlap
                genre_overlap = len(genres_i.intersection(genres_j)) / len(genres_i.union(genres_j))
                
                # Get cast and crew for both movies
                cast_i = set(movies_df.iloc[i]['cast'].split())
                cast_j = set(movies_df.iloc[j]['cast'].split())
                crew_i = set(movies_df.iloc[i]['crew'].split())
                crew_j = set(movies_df.iloc[j]['crew'].split())
                
                # Calculate cast/crew overlap
                cast_overlap = len(cast_i.intersection(cast_j)) / len(cast_i.union(cast_j)) if cast_i or cast_j else 0
                crew_overlap = len(crew_i.intersection(crew_j)) / len(crew_i.union(crew_j)) if crew_i or crew_j else 0
                
                # Calculate series similarity score
                series_similarity[i, j] = (
                    0.5 * genre_overlap +
                    0.3 * cast_overlap +
                    0.2 * crew_overlap
                )
    
    # Calculate numerical similarity
    numerical_features = ['vote_average', 'vote_count', 'runtime', 'budget', 'revenue']
    numerical_matrix = movies_df[numerical_features].values
    numerical_similarity = cosine_similarity(numerical_matrix)
    
    # Combine all similarities with adjusted weights
    final_similarity = (
        0.4 * content_similarity +  # Reduced from 0.5 to give more weight to contextual features
        0.4 * series_similarity +   # Increased from 0.3 to prioritize genre and cast/crew matches
        0.2 * numerical_similarity  # Kept the same for numerical features
    )
    
    return final_similarity

def recommend(movie, movies_df, similarity_matrix, n_recommendations=5):
    """Get movie recommendations based on selected movie."""
    try:
        # Find the index of the selected movie
        index = movies_df[movies_df['title'] == movie].index[0]
        
        # Get similarity scores for all movies
        distances = sorted(list(enumerate(similarity_matrix[index])), reverse=True, key=lambda x: x[1])
        
        recommended_movies = []
        seen_movies = set()
        
        # Get genres of the selected movie
        selected_genres = set(movies_df.iloc[index]['genres'].split())
        
        for i in distances[1:]:
            movie_data = movies_df.iloc[i[0]]
            
            # Skip if we've already seen this movie
            if movie_data['movie_id'] in seen_movies:
                continue
            
            # Get genres of the candidate movie
            candidate_genres = set(movie_data['genres'].split())
            
            # Calculate genre overlap
            genre_overlap = len(selected_genres.intersection(candidate_genres)) / len(selected_genres.union(candidate_genres))
            
            # Only include movies with significant genre overlap
            if genre_overlap > 0.3:  # At least 30% genre overlap
                # Add to seen movies
                seen_movies.add(movie_data['movie_id'])
                
                recommended_movies.append({
                    'title': movie_data['title'],
                    'genres': movie_data['genres'],
                    'rating': movie_data['vote_average'],
                    'overview': movie_data['overview'],
                    'release_date': movie_data['release_date']
                })
            
            # Stop if we have enough recommendations
            if len(recommended_movies) >= n_recommendations:
                break
        
        return recommended_movies
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return []

def train_model():
    """Train the recommendation model."""
    # Create model directory if it doesn't exist
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    
    # Load movie data
    print("Loading movie data...")
    movies = load_movie_data()
    if movies is None:
        print("Failed to load movie data")
        return
    
    # Remove duplicates based on movie_id
    movies = movies.drop_duplicates(subset=['movie_id'])
    
    # Preprocess features
    print("Preprocessing features...")
    movies = preprocess_features(movies)
    
    # Create TF-IDF vectorizer for text features
    print("Creating TF-IDF vectors...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
    
    # Calculate combined similarity matrix
    print("Calculating similarity matrix...")
    similarity = calculate_similarity_matrix(movies, tfidf_matrix)
    
    # Save the model
    print("Saving model files...")
    pickle.dump(movies, open(model_dir / 'movie_list.pkl', 'wb'))
    pickle.dump(similarity, open(model_dir / 'similarity.pkl', 'wb'))
    
    print(f"Model training completed! Total movies: {len(movies)}")

if __name__ == "__main__":
    train_model() 