# Movie Recommendation System

A content-based movie recommendation system that suggests movies based on genre similarity, cast/crew overlap, and content analysis. Built with Python, FastAPI, and Streamlit.

## Features

- Content-based recommendation using TF-IDF and cosine similarity
- Genre-based filtering for more relevant recommendations
- Cast and crew overlap consideration
- Beautiful Streamlit UI with movie posters and details
- Support for 5000+ movies from TMDB dataset

## Installation

1. Clone the repository:
```bash
git clone https://github.com/KushG-co/MOVIERECOMD.git
cd MOVIERECOMD
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
```bash
python movie_recommender/download_data.py
```

5. Train the model:
```bash
python movie_recommender/train_model.py
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run movie_recommender/app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Select a movie from the dropdown menu and click "Show Recommendations"

## Project Structure

```
movie_recommender/
├── app.py              # Streamlit web application
├── train_model.py      # Model training script
├── download_data.py    # Dataset download script
├── requirements.txt    # Project dependencies
└── model/             # Trained model files
    ├── movie_list.pkl
    └── similarity.pkl
```

## Technologies Used

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- scikit-learn
- TMDB API

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 