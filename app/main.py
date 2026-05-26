from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.schemas import RecommendationRequest, RecommendationResponse, MovieResult
from app.model import recommender


# Lifespan : Mengatur apa yang terjadi saat server start/stop
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Movie Recommendation Service starting up...")
    print("Models already loaded and ready.")
    yield
    # Shutdown
    print("Service shutting down.......")


# Inisialisasi fastAPI App

app = FastAPI(
    title="Movie Recommendation Model Service",
    description="REST API untuk rekomendasi film berbasis sinopsis menggunakan SBERT + Neural Encoder",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
# Change to spesific domain for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint 1: Health Check
# GET /

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "service": "Movie Recommendation Model Service",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "models_loaded": True,
        "total_films": 4797
    }

# Endpoint 2: Get Recommendations
# POST /recommend
# Flow:
#  1. FastAPI validasi request body otomatis via RecommendationRequest
#  2. Panggil recommender.get_recommendations()
#  3. Return RecommendationResponse

@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendation"])
def recommend_movies(request: RecommendationRequest):
    """
    Terima sinopsis → return 5 rekomendasi film paling mirip.

    - **synopsis**: Teks sinopsis film (min 10 karakter, max 2000 karakter)
    """

    try:
        results = recommender.get_recommendations(synopsis=request.synopsis, top_n=5)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {str(e)}"
        )

    movie_results = [MovieResult(**movie) for movie in results]

    return RecommendationResponse(
        status="success",
        synopsis_received=request.synopsis,
        recommendations=movie_results
    )
