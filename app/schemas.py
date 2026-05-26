from pydantic import BaseModel, Field
from typing import List

# Request Schema
# Kalau request tidak mengandung field 'synopsis', FastAPI otomatis return 422 Unprocessable Entity.

class RecommendationRequest(BaseModel):
    synopsis: str = Field(
        ...,                         
        min_length=10,                
        max_length=2000,              
        description="Sinopsis film yang ingin dicari rekomendasinya",
        examples=["A robot falls in love with a human and tries to understand emotions."]
    )

# Response Schema — satu film
# Ini representasi satu film dalam hasil rekomendasi.

class MovieResult(BaseModel):
    rank: int                        
    title: str                       
    vote_average: float               
    similarity_score: float         


# Response Schema — keseluruhan response API
# Ini yang akan diterima setelah POST request.

class RecommendationResponse(BaseModel):
    status: str                      
    synopsis_received: str            
    recommendations: List[MovieResult]  
