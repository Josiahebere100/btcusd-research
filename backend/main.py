"""BTC/USD Research Backend — FastAPI entry point."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.db import close_pool, get_pool
from app.routes import collector, research

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await get_pool()
    except Exception as e:
        print(f"[startup] DB pool init failed: {e}")
    yield
    await close_pool()


app = FastAPI(
    title="BTC/USD Research Backend",
    version="0.2.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

app.include_router(collector.router, prefix="/api/collector", tags=["collector"])
app.include_router(research.router, prefix="/api/research", tags=["research"])


@app.get("/")
async def root():
    return {"status": "ok", "service": "btcusd-research-backend"}


@app.get("/research")
async def research_page():
    """Serve the live research dashboard."""
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return JSONResponse(
            status_code=500,
            content={"error": "static/index.html not found"},
        )
    return FileResponse(index)


# Mount static assets. Note: main.py lives in backend/, so STATIC_DIR is
# backend/static/. We mount at /static.
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Safety net: /api/* never falls through to HTML.
@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "not_found", "path": request.url.path},
    )
