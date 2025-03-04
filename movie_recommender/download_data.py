import kagglehub
import os
from pathlib import Path

def download_dataset():
    """Download the TMDB movie metadata dataset using Kagglehub."""
    print("Downloading TMDB movie metadata dataset...")
    
    # Create data directory if it doesn't exist
    data_dir = Path("../data")
    data_dir.mkdir(exist_ok=True)
    
    try:
        # Download the dataset
        path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
        print("Dataset downloaded successfully!")
        print("Path to dataset files:", path)
        
        # Copy files to our data directory
        import shutil
        for file in os.listdir(path):
            if file.endswith('.csv'):
                src = os.path.join(path, file)
                dst = data_dir / file
                shutil.copy2(src, dst)
                print(f"Copied {file} to {dst}")
        
        print("\nDataset files are now in the data directory!")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_dataset() 