"""
Deal Intelligence API - Main FastAPI application
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.api.dependencies import app_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and data on startup."""
    app_state.load()
    yield


app = FastAPI(
    title="Deal Intelligence API",
    description="Revenue intelligence system for sales teams",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://frontend:80"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router, prefix="/api")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "0.1.0",
        "models_loaded": app_state.is_ready,
    }
