"""FastAPI application entry point."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.api.routes import router

app = FastAPI(
    title="Betting Market Autopsy",
    description="Rigorous analysis of prediction market efficiency",
    version="0.1.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

# Load the HTML template
TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the dashboard."""
    if TEMPLATE_PATH.exists():
        return HTMLResponse(TEMPLATE_PATH.read_text())
    return HTMLResponse("<h1>Template not found</h1>", status_code=500)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
