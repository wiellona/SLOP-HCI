from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import voice
# from app.api.v1 import sign  # Comment this out for now

app = FastAPI(
    title="SLOP ML Service API",
    description="Backend khusus AI untuk Deteksi Suara (Whisper) dan Isyarat (Siformer)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register voice router
app.include_router(voice.router, prefix="/v1/voice", tags=["Voice Audio"])

# Don't register sign router yet
# app.include_router(sign.router, prefix="/v1/sign", tags=["Sign Language"])

@app.get("/")
async def root():
    return {"message": "SLOP ML Service is running smoothly!"}

@app.get("/health")
async def health():
    return {"status": "healthy", "services": ["voice"]}