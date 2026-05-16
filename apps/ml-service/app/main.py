from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import voice 

app = FastAPI(
    title="SLOP ML Service API",
    description="Backend khusus AI untuk Deteksi Suara (Whisper) dan Isyarat (Siformer)",
    version="1.0.0"
)

# CORS Middleware agar tidak diblokir oleh browser saat di-test
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mendaftarkan endpoint voice.py (Baru voice aja, belum yang hand gesture, nanti tambahin aja di bawah sini)
app.include_router(voice.router, prefix="/v1/voice", tags=["Voice Audio"])

@app.get("/")
async def root():
    return {"message": "SLOP ML Service is running smoothly!"}