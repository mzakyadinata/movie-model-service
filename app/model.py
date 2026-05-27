import numpy as np
import pandas as pd
import keras
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Path Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

EMBEDDINGS_PATH   = os.path.join(MODELS_DIR, "embeddings.npy")
MOVIES_DATA_PATH  = os.path.join(MODELS_DIR, "movies_data.csv")
ENCODER_MODEL_PATH = os.path.join(MODELS_DIR, "encoder_model.keras")

# Class MovieRecommender
class MovieRecommender:

    def __init__(self):
        """
        Constructor — dipanggil SEKALI saat aplikasi pertama start.
        Di sini kita load semua aset yang berat ke memory.
        """
        print("Loading SBERT model...")
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

        print("Loading TF Encoder model...")
        self.encoder_model = keras.models.load_model(ENCODER_MODEL_PATH, compile=False)


        print("Loading pre-computed embeddings & movie data...")
        self.embeddings = np.load(EMBEDDINGS_PATH)           
        self.movies_df  = pd.read_csv(MOVIES_DATA_PATH)     

        print(f"Ready! {len(self.movies_df)} films loaded.")


    def encode_synopsis(self, synopsis: str) -> np.ndarray:     
        # SBERT
        sbert_vector = self.sbert_model.encode([synopsis])  # shape: (1, 384)
        return sbert_vector


    def get_recommendations(self, synopsis: str, top_n: int = 5) -> list:
        """
        STEP 3 + 4: Hitung cosine similarity dan return top-N film.

        Args:
            synopsis: teks sinopsis dari user
            top_n: jumlah rekomendasi yang diinginkan (default 5)

        Returns:
            list of dict berisi info film + similarity score
        """

        # Encode sinopsis user ke vektor 64 dimensi
        query_vector = self.encode_synopsis(synopsis)      

        # Step 3: Cosine Similarity
        similarity_scores = cosine_similarity(query_vector, self.embeddings) 
        similarity_scores = similarity_scores[0]           

        # Step 4: Ambil top-N indeks dengan skor tertinggi
        top_indices = np.argsort(similarity_scores)[::-1][:top_n]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append({
                "rank": rank,
                "title": str(self.movies_df.iloc[idx]["title"]),
                "vote_average": float(self.movies_df.iloc[idx]["vote_average"]),
                "similarity_score": round(float(similarity_scores[idx]), 4)
            })

        return results
    
# Singleton Instance
recommender = MovieRecommender()
