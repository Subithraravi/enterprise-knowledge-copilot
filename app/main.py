from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Enterprise Knowledge Copilot API",
    description="Backend API for the AI Enterprise Knowledge Copilot",
    version="1.0.0"
)

app.include_router(router)