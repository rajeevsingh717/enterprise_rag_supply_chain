"""Read-only local dashboard for article demos and system inspection."""

from __future__ import annotations

import json
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Annotated

import typer

from rag_supply_chain.health import CHECKS
from rag_supply_chain.optimization.normalization import measure_normalization

ASSET_DIR = Path(__file__).with_name("dashboard_assets")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTENT_TYPES = {".html": "text/html", ".css": "text/css", ".js": "text/javascript"}


def build_dashboard_data(
    project_root: Path = PROJECT_ROOT,
    checks: dict[str, Callable[[], tuple[bool, str]]] = CHECKS,
) -> dict:
    baseline_path = project_root / "eval/results-local.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    sample_path = project_root / "sample_docs/system_design.md"
    sample_text = sample_path.read_text(encoding="utf-8")
    normalization = measure_normalization([sample_text]).to_dict()
    services = []
    for name, check in checks.items():
        ok, detail = check()
        services.append({"name": name, "ok": ok, "detail": detail})

    qdrant_detail = next((item["detail"] for item in services if item["name"] == "Qdrant"), "")
    return {
        "mode": "sample_with_live_health",
        "services": services,
        "pipeline": [
            {"name": "Ingest", "detail": "Validated sample document", "state": "ready"},
            {"name": "Chunk", "detail": "Semantic sections + lineage", "state": "ready"},
            {"name": "Embed", "detail": qdrant_detail or "Qdrant status unavailable", "state": "ready"},
            {"name": "Retrieve", "detail": "Dense + sparse RRF", "state": "ready"},
            {"name": "Evaluate", "detail": f"{baseline['case_count']} baseline cases", "state": "ready"},
        ],
        "document": {
            "name": sample_path.name,
            "title": "System Design Notes",
            "sections": 4,
            "topics": ["Subnets", "Load Balancing", "Indexing", "Replication"],
        },
        "sample_query": {
            "label": "Illustrative sample — no live model call",
            "question": "What risk does asynchronous replication create if the primary fails?",
            "answer": (
                "Recent acknowledged writes may be lost because asynchronous replication can "
                "leave replicas behind the primary at the moment it fails."
            ),
            "citation": "System Design Notes › Databases › Replication",
        },
        "evaluation": baseline,
        "normalization": normalization,
        "telemetry_note": (
            "Token usage comes from provider responses during live runs. Cost remains unknown "
            "until current model rates are explicitly configured."
        ),
    }


class DashboardHandler(BaseHTTPRequestHandler):
    project_root = PROJECT_ROOT

    def do_GET(self) -> None:
        route = self.path.split("?", 1)[0]
        if route == "/api/dashboard":
            try:
                self._send_json(build_dashboard_data(self.project_root))
            except Exception as exc:  # noqa: BLE001 - return a useful local status error
                self._send_json({"error": str(exc)}, status=500)
            return
        asset_name = {"/": "index.html", "/styles.css": "styles.css", "/app.js": "app.js"}.get(
            route
        )
        if asset_name is None:
            self.send_error(404)
            return
        path = ASSET_DIR / asset_name
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{CONTENT_TYPES[path.suffix]}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def serve(
    host: Annotated[str, typer.Option(help="Bind address; localhost is the safe default.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8765,
) -> None:
    """Serve the read-only dashboard until Ctrl-C."""
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    typer.echo(f"Dashboard: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    typer.run(serve)


if __name__ == "__main__":
    main()
