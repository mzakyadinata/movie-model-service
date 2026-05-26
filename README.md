# Movie Recommendation Model Service

This is the model service for our capstone project — a web app that recommends movies based on a synopsis you write. This service handles all the machine learning parts, built with FastAPI and exposed as a REST API so the backend team can call it easily.

## What it does

You send a movie synopsis (a short description of a plot), and it returns 5 movies from our dataset that are most similar to that synopsis. Under the hood it runs the text through SBERT to generate embeddings, refines them with a custom-trained neural network encoder, then compares against pre-computed embeddings of 4797 movies using cosine similarity.

## Stack

- **FastAPI** — the API framework
- **Sentence Transformers (SBERT)** — converts synopsis text into vector embeddings
- **TensorFlow / Keras** — loads and runs the trained encoder model
- **scikit-learn** — cosine similarity calculation
- **Uvicorn** — ASGI server

## Project Structure

```
movie-model-service/
├── app/
│   ├── main.py        # API endpoints and app config
│   ├── model.py       # ML logic (SBERT + encoder + similarity)
│   └── schemas.py     # Request/response data shapes
├── models/
│   ├── encoder_model.keras      # Trained neural network encoder
│   ├── embeddings.npy           # Pre-computed embeddings for 4797 films
│   ├── embeddings_sbert.npy     # Raw SBERT embeddings
│   └── movies_data.csv          # Movie titles and metadata
├── Procfile
└── requirements.txt
```

## Running locally

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API

### `GET /health`

Check if the service is running.

### `POST /recommend`

Get movie recommendations based on a synopsis.

**Request body:**

```json
{
  "synopsis": "A soldier wakes up in someone else's body and discovers he's part of a mission to find the bomber of a commuter train."
}
```

**Response:**

```json
{
  "status": "success",
  "synopsis_received": "...",
  "recommendations": [
    {
      "rank": 1,
      "title": "Inception",
      "vote_average": 8.1,
      "similarity_score": 0.7823
    }
  ]
}
```

## Calling from Express

```javascript
const response = await fetch("https://your-deployed-url/recommend", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ synopsis: userSynopsis }),
});

const data = await response.json();
// data.recommendations -> array of 5 movies
```
