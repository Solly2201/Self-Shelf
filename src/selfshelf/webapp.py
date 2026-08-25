"""FastAPI application serving the dashboard and the engine's data.

Thin by design: every number it returns was computed by the frozen pricing
engine via `PricingService`. The optimization run happens once, in a
background thread at startup; endpoints return 503 with a status payload
until it finishes so the frontend can show a real loading state.
"""

import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import PricingConfig
from .webdata import PricingService

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA = REPO_ROOT / "data" / "walmart_large_sample_data_with_categories.csv"
WEB_DIR = REPO_ROOT / "web"


def create_app(
    data_path: Optional[str] = None,
    num_items: int = 50,
    config: Optional[PricingConfig] = None,
    compute_async: bool = True,
) -> FastAPI:
    service = PricingService(
        data_path=str(data_path or DEFAULT_DATA),
        num_items=num_items,
        config=config or PricingConfig(),
    )
    state = {"error": None}

    def compute():
        try:
            service.compute()
        except Exception as exc:  # surface startup failures to the UI
            state["error"] = str(exc)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not service.ready:
            if compute_async:
                threading.Thread(target=compute, daemon=True).start()
            else:
                compute()
        yield

    app = FastAPI(
        title="Self-Shelf", docs_url=None, redoc_url=None, lifespan=lifespan
    )
    app.state.service = service

    def require_ready():
        if state["error"]:
            raise HTTPException(500, detail=state["error"])
        if not service.ready:
            raise HTTPException(
                503, detail="Optimization run in progress"
            )

    @app.get("/api/status")
    def status():
        return {
            "ready": service.ready,
            "error": state["error"],
            "generated_at": service.generated_at,
        }

    @app.get("/api/dashboard")
    def dashboard():
        require_ready()
        return service.dashboard()

    @app.get("/api/products")
    def products():
        require_ready()
        return {"products": service.products(), "meta": service.meta()}

    @app.get("/api/products/{sku}")
    def product_detail(sku: str):
        require_ready()
        detail = service.product_detail(sku)
        if detail is None:
            raise HTTPException(404, detail=f"Unknown product {sku}")
        return detail

    @app.get("/api/products/{sku}/sweep")
    def product_sweep(sku: str):
        require_ready()
        sweep = service.sweep(sku)
        if sweep is None:
            raise HTTPException(404, detail=f"Unknown product {sku}")
        detail = service.product_detail(sku)
        return {
            "points": sweep,
            "current_price": detail["pricing"]["current"],
            "recommended_price": detail["pricing"]["recommended"],
            "min_allowed": detail["pricing"]["min_allowed"],
            "max_allowed": detail["pricing"]["max_allowed"],
        }

    @app.get("/api/products/{sku}/scenario")
    def product_scenario(sku: str, price: float = Query(..., gt=0)):
        require_ready()
        scenario = service.scenario(sku, price)
        if scenario is None:
            raise HTTPException(404, detail=f"Unknown product {sku}")
        return scenario

    @app.get("/api/analytics")
    def analytics():
        require_ready()
        return service.analytics()

    if WEB_DIR.exists():
        @app.get("/")
        def index():
            return FileResponse(WEB_DIR / "index.html")

        app.mount(
            "/", StaticFiles(directory=str(WEB_DIR), html=True), name="web"
        )
    else:  # pragma: no cover - only hit if the web assets are missing
        @app.get("/")
        def missing_web():
            return JSONResponse(
                {"detail": "web/ assets not found"}, status_code=500
            )

    return app
