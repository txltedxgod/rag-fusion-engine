from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
import logging

from src.core.config import settings
from src.core.exceptions import BaseAppException
from src.api.v1.router import router as api_v1_router

logging.basicConfig(level=settings.log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("rag_fusion")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing %s in %s mode...", settings.app_name, settings.app_env)
    yield
    logger.info("Gracefully shutting down %s...", settings.app_name)

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
    return response

@app.exception_handler(BaseAppException)
async def custom_app_exception_handler(request: Request, exc: BaseAppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"type": exc.__class__.__name__, "message": exc.message}}
    )

app.include_router(api_v1_router)

@app.get("/health", tags=["Monitoring"])
def health_check():
    return {"status": "healthy", "env": settings.app_env}
