import pickle
import streamlit as st
import requests
import os
from pathlib import Path
import re

# Set page config
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background-color: #2c3e50;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #34495e;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .movie-title {
        font-size: 1.4rem;
        font-weight: bold;
        text-align: center;
        margin-top: 0.5rem;
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .movie-info {
        font-size: 0.9rem;
        color: #666;
        text-align: center;
        margin-top: 0.2rem;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .movie-overview {
        font-size: 0.85rem;
        color: #444;
        text-align: center;
        margin-top: 0.2rem;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
        font-family: 'Helvetica Neue', sans-serif;
        line-height: 1.5;
    }
    .genre-tag {
        display: inline-block;
        background-color: #e8f4f8;
        padding: 0.3rem 0.8rem;
        border-radius: 1.5rem;
        margin: 0.2rem;
        font-size: 0.8rem;
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
        transition: all 0.3s ease;
    }
    .genre-tag:hover {
        background-color: #d1e8f2;
        transform: translateY(-1px);
    }
    .stSelectbox>div>div>select {
        background-color: white;
        border-radius: 5px;
        border: 1px solid #ddd;
        padding: 0.5rem;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stSelectbox>div>div>select:hover {
        border-color: #2c3e50;
    }
    .movie-card {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .rating-badge {
        display: inline-block;
        background-color: #f1c40f;
        color: #2c3e50;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    .section-title {
        color: #2c3e50;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
        font-family: 'Helvetica Neue', sans-serif;
        text-align: center;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-family: 'Helvetica Neue', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

def fetch_poster(movie_id):
    """Fetch movie poster from TMDB API."""
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        data = requests.get(url)
        data.raise_for_status()
        poster_path = data.json()['poster_path']
        return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    except Exception as e:
        st.error(f"Error fetching poster: {str(e)}")
        return None

def recommend(movie, movies_df, similarity_matrix):
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
                
                # Get poster
                poster = fetch_poster(movie_data['movie_id'])
                if poster:
                    recommended_movies.append({
                        'title': movie_data['title'],
                        'poster': poster,
                        'genres': movie_data['genres'],
                        'rating': movie_data['vote_average'],
                        'overview': movie_data['overview'],
                        'release_date': movie_data['release_date']
                    })
            
            # Stop if we have enough recommendations
            if len(recommended_movies) >= 5:
                break
        
        return recommended_movies
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return []

# Main app
st.markdown("<h1 class='section-title'>🎬 Movie Recommender System</h1>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #666; font-family: "Helvetica Neue", sans-serif; margin-bottom: 2rem;'>
        Discover your next favorite movie with our intelligent recommendation system.
        Based on genre similarity, cast/crew overlap, and content analysis.
    </div>
""", unsafe_allow_html=True)

# Load model files
try:
    model_path = Path("model")
    movies = pickle.load(open(model_path / 'movie_list.pkl', 'rb'))
    similarity = pickle.load(open(model_path / 'similarity.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model files: {str(e)}")
    st.stop()

# Movie selection with custom styling
st.markdown("<h3 style='color: #2c3e50; font-family: \"Helvetica Neue\", sans-serif;'>Select a Movie</h3>", unsafe_allow_html=True)
movie_list = sorted(movies['title'].values)
selected_movie = st.selectbox(
    "Choose a movie to get personalized recommendations",
    movie_list,
    help="Select a movie to get recommendations based on its characteristics"
)

# Show selected movie info
selected_movie_data = movies[movies['title'] == selected_movie].iloc[0]
st.markdown("<h2 class='section-title'>Selected Movie</h2>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])
with col1:
    st.image(fetch_poster(selected_movie_data['movie_id']), use_column_width=True)
with col2:
    st.markdown(f"<h3 style='color: #2c3e50; font-family: \"Helvetica Neue\", sans-serif;'>{selected_movie_data['title']}</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: #666;'>{selected_movie_data['release_date'][:4]}</p>", unsafe_allow_html=True)
    st.markdown(f"<span class='rating-badge'>⭐ {selected_movie_data['vote_average']:.1f}</span>", unsafe_allow_html=True)
    
    # Display genres as tags
    genres_html = ' '.join([f'<span class="genre-tag">{genre}</span>' for genre in selected_movie_data['genres'].split()])
    st.markdown(f"<div style='margin: 1rem 0;'>{genres_html}</div>", unsafe_allow_html=True)
    
    st.markdown(f"<p style='color: #444; line-height: 1.6;'>{selected_movie_data['overview']}</p>", unsafe_allow_html=True)

# Recommendation button
if st.button('Get Recommendations', type="primary"):
    with st.spinner('Finding the perfect movies for you...'):
        recommended_movies = recommend(selected_movie, movies, similarity)
        
        if recommended_movies:
            st.markdown("<h2 class='section-title'>Recommended Movies</h2>", unsafe_allow_html=True)
            
            # Create columns for recommendations
            cols = st.columns(5)
            
            # Display recommendations
            for i, (col, movie) in enumerate(zip(cols, recommended_movies)):
                with col:
                    st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
                    st.image(movie['poster'], use_column_width=True)
                    st.markdown(f"<p class='movie-title'>{movie['title']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<span class='rating-badge'>⭐ {movie['rating']:.1f}</span>", unsafe_allow_html=True)
                    
                    # Display genres as tags
                    genres_html = ' '.join([f'<span class="genre-tag">{genre}</span>' for genre in movie['genres'].split()])
                    st.markdown(f"<div style='margin: 0.5rem 0;'>{genres_html}</div>", unsafe_allow_html=True)
                    
                    st.markdown(f"<p class='movie-overview'>{movie['overview']}</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("No recommendations found. Please try another movie.")

# Footer
st.markdown("""
    <div class='footer'>
        <p>Built with ❤️ using Streamlit</p>
        <p>Data source: TMDB API</p>
        <p style='font-size: 0.8rem; margin-top: 1rem;'>© 2024 Movie Recommender System</p>
    </div>
""", unsafe_allow_html=True) 