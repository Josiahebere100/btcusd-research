"""BTC/USD Research Backend — FastAPI entry point."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.db import close_pool, get_pool
from app.routes import collector, dashboard, memory, research, signals

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
    version="0.3.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

app.include_router(collector.router, prefix="/api/collector", tags=["collector"])
app.include_router(research.router, prefix="/api/research", tags=["research"])
app.include_router(signals.router, prefix="/api/signals", tags=["signals"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])
app.include_router(memory.router, prefix="/api/memory", tags=["memory"])


@app.get("/")
async def root():
    return {"status": "ok", "service": "btcusd-research-backend"}


@app.get("/research")
async def research_page():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return JSONResponse(
            status_code=500,
            content={"error": "static/index.html not found"},
        )
    return FileResponse(index)


@app.get("/signals")
async def signals_page():
    page = STATIC_DIR / "signals.html"
    if not page.exists():
        return JSONResponse(
            status_code=500,
            content={"error": "static/signals.html not found"},
        )
    return FileResponse(page)

@app.get("/dashboard")
async def dashboard_page():
    page = STATIC_DIR / "dashboard.html"
    if not page.exists():
        return JSONResponse(
            status_code=500,
            content={"error": "static/dashboard.html not found"},
        )
    return FileResponse(page)

@app.get("/memory")
async def memory_page():
    page = STATIC_DIR / "memory.html"
    if not page.exists():
        return JSONResponse(
            status_code=500,
            content={"error": "static/memory.html not found"},
        )
    return FileResponse(page)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "not_found", "path": request.url.path},
    )
