from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import voice
from app.api.v1 import sign
from loguru import logger

app = FastAPI(
    title="SLOP ML Service API",
    description="Tuhan, aku capek",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(voice.router, prefix="/v1/voice", tags=["Voice Audio"])
app.include_router(sign.router, prefix="/v1/sign", tags=["Sign Language"])

@app.on_event("startup")
async def startup_event():
    logger.info("Starting SLOP ML Service...")
    logger.info("Voice service ready")
    logger.info("Sign language service ready")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down SLOP ML Service...")

@app.get("/")
async def root():
    return {
        "message": "SLOP ML Service is running smoothly!",
        "services": ["voice", "sign"]
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "services": ["voice", "sign"]
    }