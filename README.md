# 🎬 Movie Recommendation Model Service

REST API berbasis FastAPI untuk rekomendasi film berdasarkan sinopsis,
menggunakan SBERT + Neural Encoder (TensorFlow/Keras).

---

## 📁 Struktur Project

```
movie-model-service/
├── app/
│   ├── __init__.py      # penanda Python package
│   ├── main.py          # FastAPI app & endpoint definitions
│   ├── model.py         # ML logic: SBERT + encoder + cosine similarity
│   └── schemas.py       # Pydantic request/response schemas
├── models/              # ← TARUH FILE ML DI SINI
│   ├── embeddings.npy
│   ├── movies_data.csv
│   └── encoder_model.keras
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup (Pertama Kali)

### 1. Pastikan Python 3.10+ terinstall

```bash
python --version
```

### 2. Buat virtual environment (sangat disarankan)

```bash
# Buat venv
python -m venv venv

# Aktifkan — Windows:
venv\Scripts\activate

# Aktifkan — Mac/Linux:
source venv/bin/activate
```

### 3. Install semua dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ Proses ini butuh waktu ~5-10 menit karena TensorFlow dan
> sentence-transformers cukup besar. Pastikan koneksi internet stabil.

### 4. Taruh file ML ke folder /models

Salin file-file berikut ke dalam folder `models/`:

- `embeddings.npy`
- `movies_data.csv`
- `encoder_model.keras`

---

## Menjalankan Server

Dari root folder project (tempat README.md ini berada):

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Penjelasan flag:

- `app.main:app` — "di folder app, file main.py, object bernama app"
- `--reload` — server auto-restart kalau ada perubahan kode (mode development)
- `--host 0.0.0.0` — bisa diakses dari jaringan lokal (tidak hanya localhost)
- `--port 8000` — port yang digunakan

Server berhasil jalan kalau muncul:

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
✅ Ready! 4797 films loaded.
```

---

## 🧪 Testing di Postman

### 1. Health Check

- **Method**: GET
- **URL**: `http://localhost:8000/health`
- **Expected Response**:

```json
{
  "status": "healthy",
  "models_loaded": true,
  "total_films": 4797
}
```

### 2. Get Recommendations

- **Method**: POST
- **URL**: `http://localhost:8000/recommend`
- **Headers**: `Content-Type: application/json`
- **Body (raw JSON)**:

```json
{
  "synopsis": "A space explorer crash-lands on an alien planet and must find a way to survive while uncovering the secrets of an ancient civilization."
}
```

- **Expected Response**:

```json
{
    "status": "success",
    "synopsis_received": "A space explorer crash-lands on an alien planet...",
    "recommendations": [
        {
            "rank": 1,
            "title": "Interstellar",
            "vote_average": 8.1,
            "similarity_score": 0.9234
        },
        ...
    ]
}
```

### 3. Interactive API Docs (Swagger UI)

FastAPI otomatis generate dokumentasi interaktif di:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

Kamu bisa langsung test endpoint dari browser tanpa Postman!

---

## 🔌 Integrasi dengan Express Backend

Dari Express (Node.js), hit endpoint ini dengan:

```javascript
const response = await fetch("http://localhost:8000/recommend", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ synopsis: userSynopsis }),
});
const data = await response.json();
// data.recommendations = array 5 film
```

---

## ❓ Troubleshooting

| Error                               | Kemungkinan Penyebab                | Solusi                            |
| ----------------------------------- | ----------------------------------- | --------------------------------- |
| `ModuleNotFoundError`               | Dependencies belum terinstall       | `pip install -r requirements.txt` |
| `FileNotFoundError: embeddings.npy` | File ML belum dipindah ke `/models` | Salin file ke folder `models/`    |
| `OSError: Unable to open file`      | File .keras corrupt/salah versi     | Minta file ulang ke divisi ML     |
| Port 8000 sudah dipakai             | Ada proses lain di port itu         | Ganti ke `--port 8001`            |
