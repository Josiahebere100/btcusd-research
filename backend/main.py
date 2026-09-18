"""BTC/USD Research Backend — FastAPI entry point."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.db import close_pool, get_pool
from app.routes import collector


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: warm the DB pool
    try:
        await get_pool()
    except Exception as e:
        # Do not crash — health route will report the error.
        print(f"[startup] DB pool init failed: {e}")
    yield
    # Shutdown: close pool
    await close_pool()


app = FastAPI(
    title="BTC/USD Research Backend",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

# Mount the collector routes
app.include_router(collector.router, prefix="/api/collector", tags=["collector"])


@app.get("/")
async def root():
    return {"status": "ok", "service": "btcusd-research-backend"}


# Safety net: ensure /api/* NEVER falls back to HTML for unknown routes.
@app.exception_handler(404)
async def not_found(request: Request, exc):
    if request.url.path.startswith("/api/"):
        return JSONResponse(
            status_code=404,
            content={"error": "not_found", "path": request.url.path},
        )
    return JSONResponse(status_code=404, content={"error": "not_found"})