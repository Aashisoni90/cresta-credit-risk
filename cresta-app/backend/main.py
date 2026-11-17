"""
CRESTA Backend — FastAPI Application (real data, Phase 4)
"""
import hashlib
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

import config
from ml.model_manager import load_models, load_test_data
from services import real_data
from api import (dashboard, companies, prediction, model_eval, explainability,
                 threshold, data_quality, features, export)


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    load_models()          # panel, matrix, universe, model; scores every row
    load_test_data()
    # TreeSHAP costs ~2.5s to construct. Paying that at startup rather than on
    # an analyst's first click into a company is the difference between a warm
    # service and one that stalls on the most important interaction it has.
    real_data.warm_explainer()
    print(f"[startup] ready in {time.perf_counter() - t0:.2f}s")
    yield


app = FastAPI(
    title="CRESTA API",
    description="Explainable corporate distress early warning — Indian listed equities",
    version=config.MODEL_A_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # Local dev: the Next dev server may land on any free port (3000, 3001, …).
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The model-performance and data-quality payloads are large and highly
# compressible JSON (curve arrays, threshold sweeps, the model card).
app.add_middleware(GZipMiddleware, minimum_size=1024)


# ── caching ───────────────────────────────────────────────────────────
# Everything this API serves is derived from artefacts fixed at a model
# version: the matrix is scored once at startup and never mutates while the
# process runs. So responses are safely cacheable FOR THIS VERSION, and the
# ETag is version-derived — deploying a new model version invalidates every
# cached response automatically, which is the property that matters. A
# time-based TTL alone would keep serving v1.1.0 numbers after v1.2.0 shipped.
#
# Per-company predictions get a shorter max-age than the static report
# endpoints only so a UI refresh picks up a restart promptly; correctness does
# not depend on it, because the ETag already pins the version.
_STATIC_PREFIXES = ("/api/model/", "/api/features", "/api/data-quality",
                    "/api/threshold", "/api/companies/sectors")
_NO_STORE = ("/api/health",)
_NEVER_CACHE_PREFIX = ("/api/export/",)


@app.middleware("http")
async def cache_headers(request: Request, call_next):
    started = time.perf_counter()
    response: Response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.1f}"

    path = request.url.path
    # HEAD is listed so that it caches identically to GET IF it is ever
    # registered. Today FastAPI's @router.get does not register HEAD, so a
    # HEAD request 405s and falls into the no-store branch below on status
    # alone. Noted rather than worked around: nothing in the product issues
    # HEAD, and faking support would hide the 405 from whoever eventually does.
    if request.method not in ("GET", "HEAD") or response.status_code >= 400:
        response.headers.setdefault("Cache-Control", "no-store")
        return response
    if path in _NO_STORE or path.startswith(_NEVER_CACHE_PREFIX):
        response.headers["Cache-Control"] = "no-store"
        return response

    version = config.MODEL_A_VERSION
    response.headers["X-Model-Version"] = version
    max_age = 3600 if path.startswith(_STATIC_PREFIXES) else 300
    response.headers["Cache-Control"] = f"public, max-age={max_age}"
    etag = hashlib.sha256(
        f"{version}|{path}|{request.url.query}".encode()).hexdigest()[:32]
    response.headers["ETag"] = f'W/"{etag}"'
    return response


# Register routers
app.include_router(dashboard.router, prefix="/api", tags=["Dashboard"])
app.include_router(companies.router, prefix="/api", tags=["Companies"])
app.include_router(prediction.router, prefix="/api", tags=["Prediction"])
app.include_router(model_eval.router, prefix="/api", tags=["Model Evaluation"])
app.include_router(explainability.router, prefix="/api", tags=["Explainability"])
app.include_router(threshold.router, prefix="/api", tags=["Threshold"])
app.include_router(data_quality.router, prefix="/api", tags=["Data Quality"])
app.include_router(features.router, prefix="/api", tags=["Features"])
app.include_router(export.router, prefix="/api", tags=["Export"])


@app.get("/api/health")
async def health():
    s = real_data.state()
    return {"status": "ok", "service": "cresta-api",
            "model_version": s["version"], "n_scored_rows": int(len(s["matrix"]))}
