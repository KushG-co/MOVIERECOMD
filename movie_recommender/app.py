import pickle
import streamlit as st
import requests
import os
from pathlib import Path
import re

# Set page config
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .movie-title {
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        margin-top: 0.5rem;
    }
    .movie-info {
        font-size: 0.9rem;
        color: #666;
        text-align: center;
        margin-top: 0.2rem;
    }
    .movie-overview {
        font-size: 0.8rem;
        color: #444;
        text-align: center;
        margin-top: 0.2rem;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .genre-tag {
        display: inline-block;
        background-color: #e0e0e0;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        margin: 0.1rem;
        font-size: 0.8rem;
        color: #666;
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
st.title("🎬 Movie Recommender System")
st.markdown("""
    This app recommends movies based on your selection, considering:
    - Genre similarity
    - Cast and crew overlap
    - Content similarity
    - Ratings and popularity
""")

# Load model files
try:
    model_path = Path("model")
    movies = pickle.load(open(model_path / 'movie_list.pkl', 'rb'))
    similarity = pickle.load(open(model_path / 'similarity.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model files: {str(e)}")
    st.stop()

# Movie selection
movie_list = sorted(movies['title'].values)
selected_movie = st.selectbox(
    "Select a movie from the dropdown",
    movie_list,
    help="Choose a movie to get recommendations"
)

# Show selected movie info
selected_movie_data = movies[movies['title'] == selected_movie].iloc[0]
st.subheader("Selected Movie")
col1, col2 = st.columns([1, 3])
with col1:
    st.image(fetch_poster(selected_movie_data['movie_id']), use_column_width=True)
with col2:
    st.markdown(f"**{selected_movie_data['title']}** ({selected_movie_data['release_date'][:4]})")
    st.markdown(f"Rating: {selected_movie_data['vote_average']:.1f} ⭐")
    
    # Display genres as tags
    genres_html = ' '.join([f'<span class="genre-tag">{genre}</span>' for genre in selected_movie_data['genres'].split()])
    st.markdown(f"**Genres:** {genres_html}", unsafe_allow_html=True)
    
    st.markdown(f"**Overview:** {selected_movie_data['overview']}")

# Recommendation button
if st.button('Show Recommendations', type="primary"):
    with st.spinner('Getting recommendations...'):
        recommended_movies = recommend(selected_movie, movies, similarity)
        
        if recommended_movies:
            st.subheader("Recommended Movies")
            
            # Create columns for recommendations
            cols = st.columns(5)
            
            # Display recommendations
            for i, (col, movie) in enumerate(zip(cols, recommended_movies)):
                with col:
                    st.image(movie['poster'], use_column_width=True)
                    st.markdown(f"<p class='movie-title'>{movie['title']}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p class='movie-info'>Rating: {movie['rating']:.1f} ⭐</p>", unsafe_allow_html=True)
                    
                    # Display genres as tags
                    genres_html = ' '.join([f'<span class="genre-tag">{genre}</span>' for genre in movie['genres'].split()])
                    st.markdown(f"<p class='movie-info'>{genres_html}</p>", unsafe_allow_html=True)
                    
                    st.markdown(f"<p class='movie-overview'>{movie['overview']}</p>", unsafe_allow_html=True)
        else:
            st.warning("No recommendations found. Please try another movie.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit</p>
        <p>Data source: TMDB API</p>
    </div>
""", unsafe_allow_html=True) 