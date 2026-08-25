"""FastAPI application serving the dashboard and the engine's data.

Thin by design: every number it returns was computed by the frozen pricing
engine via a ``BasePricingService``. Two data sources exist side by side —

    synthetic: the demo simulator pipeline (always available)
    custom:    a user-imported dataset (products + transaction history)

— and the active source is explicit and persisted; synthetic and custom
data are never mixed. The optimization run happens in a background thread
at startup; endpoints return 503 with a status payload until it finishes so
the frontend can show a real loading state.
"""

import io
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import (
    Body, FastAPI, File, HTTPException, Query, UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import PricingConfig
from .customdata import (
    CustomDataset,
    build_meta,
    fields_for,
    load_dataset,
    save_dataset,
    suggest_mapping,
    validate_products,
    validate_transactions,
)
from .webdata import CustomPricingService, PricingService

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA = REPO_ROOT / "data" / "walmart_large_sample_data_with_categories.csv"
DEFAULT_CUSTOM_DIR = REPO_ROOT / "data" / "custom"
WEB_DIR = REPO_ROOT / "web"

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def create_app(
    data_path: Optional[str] = None,
    num_items: int = 50,
    config: Optional[PricingConfig] = None,
    compute_async: bool = True,
    custom_dir: Optional[str] = None,
) -> FastAPI:
    config = config or PricingConfig()
    custom_path = Path(custom_dir or DEFAULT_CUSTOM_DIR)
    uploads_path = custom_path / "uploads"
    state_file = custom_path / "state.json"

    services: Dict[str, object] = {
        "synthetic": PricingService(
            data_path=str(data_path or DEFAULT_DATA),
            num_items=num_items,
            config=config,
        ),
        "custom": None,
    }
    state = {
        "active": "synthetic",
        "errors": {"synthetic": None, "custom": None},
    }
    lock = threading.Lock()

    # -- source management ---------------------------------------------------

    def load_state():
        if state_file.exists():
            try:
                with open(state_file, encoding="utf-8") as fh:
                    saved = json.load(fh)
                if saved.get("active_source") in ("synthetic", "custom"):
                    state["active"] = saved["active_source"]
            except (OSError, ValueError):
                pass

    def save_state():
        custom_path.mkdir(parents=True, exist_ok=True)
        with open(state_file, "w", encoding="utf-8") as fh:
            json.dump({"active_source": state["active"]}, fh)

    def compute_synthetic():
        try:
            services["synthetic"].compute()
        except Exception as exc:  # surface startup failures to the UI
            state["errors"]["synthetic"] = str(exc)

    def compute_custom() -> bool:
        """(Re)build the custom service from the persisted dataset."""
        dataset = load_dataset(str(custom_path))
        if dataset is None:
            state["errors"]["custom"] = None
            services["custom"] = None
            return False
        try:
            service = CustomPricingService(
                str(custom_path), config=config, dataset=dataset
            )
            service.compute()
            services["custom"] = service
            state["errors"]["custom"] = None
            return True
        except Exception as exc:
            state["errors"]["custom"] = str(exc)
            services["custom"] = None
            return False

    def startup():
        load_state()
        compute_synthetic()
        if (custom_path / "products.csv").exists():
            compute_custom()
        if state["active"] == "custom" and services["custom"] is None:
            # Persisted preference points at data that no longer loads.
            state["active"] = "synthetic"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not services["synthetic"].ready:
            if compute_async:
                threading.Thread(target=startup, daemon=True).start()
            else:
                startup()
        yield

    app = FastAPI(
        title="Self-Shelf", docs_url=None, redoc_url=None, lifespan=lifespan
    )
    app.state.services = services

    def active_service():
        return services[state["active"]]

    def require_ready():
        service = active_service()
        error = state["errors"][state["active"]]
        if error:
            raise HTTPException(500, detail=error)
        if service is None:
            raise HTTPException(
                503, detail="the active data source is not available"
            )
        if not service.ready:
            raise HTTPException(503, detail="Optimization run in progress")
        return service

    # -- status --------------------------------------------------------------

    @app.get("/api/status")
    def status():
        service = active_service()
        custom = services["custom"]
        return {
            "ready": bool(service is not None and service.ready),
            "error": state["errors"][state["active"]],
            "generated_at": (
                service.generated_at if service is not None else None
            ),
            "source": state["active"],
            "synthetic": {"ready": services["synthetic"].ready},
            "custom": {
                "available": custom is not None,
                "ready": bool(custom is not None and custom.ready),
                "error": state["errors"]["custom"],
            },
        }

    # -- core views (served from the ACTIVE source) --------------------------

    @app.get("/api/dashboard")
    def dashboard():
        return require_ready().dashboard()

    @app.get("/api/products")
    def products():
        service = require_ready()
        return {"products": service.products(), "meta": service.meta()}

    @app.get("/api/products/{sku}")
    def product_detail(sku: str):
        detail = require_ready().product_detail(sku)
        if detail is None:
            raise HTTPException(404, detail=f"Unknown product {sku}")
        return detail

    @app.get("/api/products/{sku}/sweep")
    def product_sweep(sku: str):
        service = require_ready()
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
        scenario = require_ready().scenario(sku, price)
        if scenario is None:
            raise HTTPException(404, detail=f"Unknown product {sku}")
        return scenario

    # -- multi-period paths --------------------------------------------------

    @app.get("/api/products/{sku}/path")
    def product_path(sku: str):
        payload = require_ready().path(sku)
        if payload is None:
            raise HTTPException(404, detail=f"Unknown product {sku}")
        return payload

    @app.post("/api/products/{sku}/path/scenario")
    def product_path_scenario(
        sku: str,
        daily_prices: List[float] = Body(..., embed=True),
    ):
        if len(daily_prices) > 60:
            raise HTTPException(
                422, detail="a path may contain at most 60 days"
            )
        result = require_ready().path_scenario(sku, daily_prices)
        if result is None:
            raise HTTPException(404, detail=f"Unknown product {sku}")
        return result

    @app.get("/api/analytics")
    def analytics():
        return require_ready().analytics()

    # -- exports -------------------------------------------------------------

    def _csv_response(text: str, filename: str) -> Response:
        return Response(
            content=text,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            },
        )

    @app.get("/api/export/recommendations.csv")
    def export_recommendations():
        service = require_ready()
        return _csv_response(
            service.export_recommendations_csv(),
            f"selfshelf_recommendations_{service.source}.csv",
        )

    @app.get("/api/export/paths.csv")
    def export_paths():
        service = require_ready()
        return _csv_response(
            service.export_paths_csv(),
            f"selfshelf_price_paths_{service.source}.csv",
        )

    # -- custom data import --------------------------------------------------

    def _upload_file(kind: str) -> Path:
        return uploads_path / f"{kind}.csv"

    def _read_upload(kind: str) -> pd.DataFrame:
        path = _upload_file(kind)
        if not path.exists():
            raise HTTPException(
                400, detail=f"no {kind} file has been uploaded yet"
            )
        try:
            return pd.read_csv(path)
        except Exception as exc:
            raise HTTPException(
                400, detail=f"could not parse the {kind} CSV: {exc}"
            )

    @app.post("/api/data/upload")
    async def data_upload(
        kind: str = Query(..., pattern="^(products|transactions)$"),
        file: UploadFile = File(...),
    ):
        raw = await file.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, detail="file exceeds the 25 MB limit")
        try:
            frame = pd.read_csv(io.BytesIO(raw))
        except Exception as exc:
            raise HTTPException(
                400, detail=f"could not parse the CSV: {exc}"
            )
        if frame.empty or not len(frame.columns):
            raise HTTPException(400, detail="the CSV contains no data rows")

        uploads_path.mkdir(parents=True, exist_ok=True)
        with open(_upload_file(kind), "wb") as fh:
            fh.write(raw)

        required, optional = fields_for(kind)
        sample = frame.head(5).where(frame.head(5).notna(), None)
        return {
            "kind": kind,
            "filename": file.filename,
            "columns": [str(c) for c in frame.columns],
            "row_count": int(len(frame)),
            "sample_rows": sample.astype(object).to_dict(orient="records"),
            "suggested_mapping": suggest_mapping(frame.columns, kind),
            "required_fields": required,
            "optional_fields": optional,
        }

    def _validate_both(
        products_mapping: Dict[str, str],
        transactions_mapping: Dict[str, str],
    ):
        products_result = validate_products(
            _read_upload("products"), products_mapping
        )
        known = (
            products_result.valid["product_id"]
            if products_result.ok and len(products_result.valid)
            else []
        )
        transactions_result = validate_transactions(
            _read_upload("transactions"), transactions_mapping,
            known_ids=list(known),
        )
        return products_result, transactions_result

    def _preview(frame: pd.DataFrame, limit: int = 8):
        head = frame.head(limit)
        if "date" in head.columns:
            head = head.assign(date=head["date"].astype(str))
        return head.where(head.notna(), None).astype(object).to_dict(
            orient="records"
        )

    def _persist_rejected(products_result, transactions_result):
        custom_path.mkdir(parents=True, exist_ok=True)
        for kind, result in (
            ("products", products_result),
            ("transactions", transactions_result),
        ):
            rejected_file = custom_path / f"rejected_{kind}.csv"
            if len(result.rejected):
                result.rejected.to_csv(rejected_file, index=False)
            elif rejected_file.exists():
                rejected_file.unlink()

    @app.post("/api/data/validate")
    def data_validate(
        products_mapping: Dict[str, str] = Body(...),
        transactions_mapping: Dict[str, str] = Body(...),
    ):
        products_result, transactions_result = _validate_both(
            products_mapping, transactions_mapping
        )
        _persist_rejected(products_result, transactions_result)
        return {
            "products": {
                **products_result.summary(),
                "preview": _preview(products_result.valid),
            },
            "transactions": {
                **transactions_result.summary(),
                "preview": _preview(transactions_result.valid),
            },
            "can_import": bool(
                products_result.ok
                and transactions_result.ok
                and len(products_result.valid)
                and len(transactions_result.valid)
            ),
        }

    @app.post("/api/data/import")
    def data_import(
        products_mapping: Dict[str, str] = Body(...),
        transactions_mapping: Dict[str, str] = Body(...),
    ):
        with lock:
            products_result, transactions_result = _validate_both(
                products_mapping, transactions_mapping
            )
            if not products_result.ok or not len(products_result.valid):
                raise HTTPException(400, detail={
                    "message": "the products file cannot be imported",
                    "errors": products_result.errors or [
                        "no valid product rows"
                    ],
                })
            if not transactions_result.ok or not len(
                transactions_result.valid
            ):
                raise HTTPException(400, detail={
                    "message":
                        "the transactions file cannot be imported — "
                        "Self-Shelf needs sales history to estimate demand",
                    "errors": transactions_result.errors or [
                        "no valid transaction rows"
                    ],
                })

            dataset = CustomDataset(
                products_result.valid,
                transactions_result.valid,
                build_meta(
                    products_result, transactions_result,
                    source_files={
                        "products": _upload_file("products").name,
                        "transactions": _upload_file("transactions").name,
                    },
                    mappings={
                        "products": products_mapping,
                        "transactions": transactions_mapping,
                    },
                ),
            )
            save_dataset(dataset, str(custom_path))
            _persist_rejected(products_result, transactions_result)

            if not compute_custom():
                raise HTTPException(400, detail={
                    "message": "the dataset could not be optimized",
                    "errors": [state["errors"]["custom"]],
                })
            state["active"] = "custom"
            save_state()

        service = services["custom"]
        return {
            "imported": True,
            "source": "custom",
            "quality": service.meta()["quality"],
            "products": {
                **products_result.summary(),
            },
            "transactions": {
                **transactions_result.summary(),
            },
            "elasticities": service.meta()["elasticities"],
        }

    @app.post("/api/data/source")
    def data_source(source: str = Body(..., embed=True)):
        if source not in ("synthetic", "custom"):
            raise HTTPException(422, detail="unknown source")
        with lock:
            if source == "custom" and services["custom"] is None:
                compute_custom()
                if services["custom"] is None:
                    raise HTTPException(400, detail=(
                        state["errors"]["custom"]
                        or "no custom dataset has been imported"
                    ))
            state["active"] = source
            save_state()
        return status()

    @app.get("/api/data/status")
    def data_status():
        custom = services["custom"]
        meta = custom.meta() if custom is not None and custom.ready else None
        rejected = {
            kind: (custom_path / f"rejected_{kind}.csv").exists()
            for kind in ("products", "transactions")
        }
        return {
            "active_source": state["active"],
            "custom": {
                "available": custom is not None,
                "error": state["errors"]["custom"],
                "meta": meta,
                "rejected_files": rejected,
            },
            "uploads": {
                kind: _upload_file(kind).exists()
                for kind in ("products", "transactions")
            },
        }

    @app.get("/api/data/rejected")
    def data_rejected(
        kind: str = Query(..., pattern="^(products|transactions)$"),
    ):
        path = custom_path / f"rejected_{kind}.csv"
        if not path.exists():
            raise HTTPException(
                404, detail=f"no rejected {kind} rows from the last import"
            )
        return _csv_response(
            path.read_text(encoding="utf-8"), f"rejected_{kind}.csv"
        )

    # -- static frontend -----------------------------------------------------

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
