# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local/Tailscale authoring dashboard. Run with --host and --port to serve it."""

import argparse
import asyncio
import base64
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, closing, contextmanager
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import threading
import time
from typing import Any, BinaryIO, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
import uvicorn

from dimos.constants import DIMOS_PROJECT_ROOT
from dimos.evals.review.habitat.applied_review import current_reviews
from dimos.evals.review.habitat.catalog import build_catalog


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int = Field(default=0, ge=0)
    status: Literal["unreviewed", "approved", "changes", "remove"] = "unreviewed"
    comment: str = Field(default="", max_length=20000)
    proposed_question: str = Field(default="", max_length=10000)
    proposed_options: str = Field(default="", max_length=10000)
    proposed_answer: str = Field(default="", max_length=10000)
    camera: dict[str, Any] | None = None


class Control(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    scene: str
    floor: int = Field(default=0, ge=0, le=50)
    cut_height: float = Field(default=1.8, ge=0.3, le=5)
    mode: Literal["topdown", "orbit"] = "topdown"
    cutaway: bool = True
    orbit_yaw: float = Field(default=35, ge=-360, le=360)
    orbit_elevation: float = Field(default=60, ge=10, le=85)
    orbit_zoom: float = Field(default=1, ge=0.25, le=24)
    pan_x: float = Field(default=0, ge=-1000, le=1000)
    pan_z: float = Field(default=0, ge=-1000, le=1000)


class FeedbackStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS reviews (scene TEXT, question TEXT, revision INTEGER, data TEXT, PRIMARY KEY(scene,question))"
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path, timeout=15)) as db, db:
            yield db

    def all(self) -> dict[str, Any]:
        with self.connect() as db:
            result: dict[str, Any] = {}
            for scene, question, data in db.execute("SELECT scene,question,data FROM reviews"):
                result.setdefault(scene, {})[question] = json.loads(data)
            return result

    def save(
        self, scene: str, question: str, review: Review, original: dict[str, Any]
    ) -> dict[str, Any]:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT revision FROM reviews WHERE scene=? AND question=?", (scene, question)
            ).fetchone()
            revision = row[0] if row else 0
            if review.revision != revision:
                raise HTTPException(
                    409,
                    "This review changed in another tab. Reload to see the saved version; copy your edits first.",
                )
            data = review.model_dump() | {
                "revision": revision + 1,
                "updated_at": time.time(),
                "original": original,
            }
            db.execute(
                "INSERT OR REPLACE INTO reviews VALUES (?,?,?,?)",
                (scene, question, revision + 1, json.dumps(data)),
            )
            return data


class Worker:
    """Serialize camera/load requests over a private socket to the renderer process."""

    def __init__(self, python: Path, log_path: Path) -> None:
        self.python, self.log_path = python, log_path
        self.lock = threading.Lock()
        self.process: subprocess.Popen[bytes] | None = None
        self.stream: BinaryIO | None = None
        self.sock: socket.socket | None = None
        self.current: dict[str, Any] | None = None

    def start(self) -> None:
        parent, child = socket.socketpair()
        parent.settimeout(180)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ | {
            "MAGNUM_LOG": "quiet",
            "HABITAT_SIM_LOG": "quiet",
            "PYTHONUNBUFFERED": "1",
        }
        try:
            with self.log_path.open("ab", buffering=0) as log:
                self.process = subprocess.Popen(
                    [
                        str(self.python),
                        str(Path(__file__).with_name("worker.py")),
                        str(child.fileno()),
                    ],
                    pass_fds=(child.fileno(),),
                    stdout=log,
                    stderr=log,
                    env=env,
                )
        except Exception:
            parent.close()
            raise
        finally:
            child.close()
        self.sock = parent
        self.stream = parent.makefile("rwb")

    def stop(self) -> None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.stream:
            self.stream.close()
        if self.sock:
            self.sock.close()
        self.process = self.stream = self.sock = self.current = None

    def request(self, cmd: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                self.stop()
                self.start()
            if cmd["action"] == "load":
                self.current = None
            try:
                assert self.stream is not None
                self.stream.write(json.dumps(cmd).encode() + b"\n")
                self.stream.flush()
                line = self.stream.readline(8 * 1024 * 1024)
                if not line:
                    raise RuntimeError(f"Renderer exited; see {self.log_path}")
                result = json.loads(line)
            except Exception:
                self.stop()
                raise
            if not result["ok"]:
                raise RuntimeError(result["error"])
            self.current = result["state"]
            return result


def create_app(catalog: dict[str, Any], reviews: FeedbackStore, worker: Worker) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await asyncio.to_thread(worker.stop)

    app = FastAPI(lifespan=lifespan)
    app.state.loading = False

    def scene(slug: str) -> dict[str, Any]:
        if slug not in catalog:
            raise HTTPException(404, "Unknown scene")
        return catalog[slug]

    def check_origin(origin: str | None, host: str) -> None:
        if origin is not None and origin not in {f"http://{host}", f"https://{host}"}:
            raise HTTPException(403, "Cross-origin control requests are not allowed")

    @app.middleware("http")
    async def same_origin(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.method not in {"GET", "HEAD"}:
            try:
                check_origin(request.headers.get("origin"), request.headers.get("host", ""))
            except HTTPException as exc:
                return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(Path(__file__).with_name("index.html"))

    @app.get("/api/catalog")
    def get_catalog() -> dict[str, Any]:
        return {
            "scenes": list(catalog.values()),
            "reviews": current_reviews(catalog, reviews.all()),
            "feedback_path": str(reviews.path),
        }

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return {"loading": app.state.loading, "camera": worker.current}

    @app.post("/api/load/{slug}")
    async def load(slug: str) -> dict[str, Any]:
        config = scene(slug)
        if app.state.loading:
            raise HTTPException(409, "Another scene is loading; try again shortly")
        app.state.loading = True
        try:
            result = await asyncio.to_thread(worker.request, {"action": "load", "scene": config})
            return result
        except Exception as exc:
            raise HTTPException(503, str(exc)) from exc
        finally:
            app.state.loading = False

    @app.websocket("/api/camera")
    async def camera(ws: WebSocket) -> None:
        try:
            check_origin(ws.headers.get("origin"), ws.headers.get("host", ""))
        except HTTPException:
            await ws.close(code=1008)
            return
        await ws.accept()
        try:
            while True:
                raw = await ws.receive_json()
                try:
                    cmd = Control.model_validate(raw)
                    scene(cmd.scene)
                    if app.state.loading:
                        raise ValueError("Scene is loading")
                    result = await asyncio.to_thread(
                        worker.request, {"action": "control", **cmd.model_dump()}
                    )
                    await ws.send_json({"state": result["state"]})
                    await ws.send_bytes(base64.b64decode(result["jpeg"]))
                except Exception as exc:
                    await ws.send_json({"error": str(exc)})
        except WebSocketDisconnect:
            pass

    @app.put("/api/reviews/{slug}/{qid}")
    def save(slug: str, qid: str, review: Review) -> dict[str, Any]:
        config = scene(slug)
        original = next((q for q in config["questions"] if q["id"] == qid), None)
        if original is None:
            original = reviews.all().get(slug, {}).get(qid, {}).get("original")
        if original is None:
            raise HTTPException(404, "Unknown question")
        return reviews.save(slug, qid, review, original)

    @app.post("/api/questions/{slug}")
    def add(slug: str, review: Review) -> dict[str, Any]:
        scene(slug)
        qid = "new-" + str(uuid4())
        original = {
            "id": qid,
            "new": True,
            "question": "New question",
            "options": "",
            "reference": {"answer": ""},
            "tags": [],
        }
        return {"id": qid, "review": reviews.save(slug, qid, review, original)}

    @app.get("/api/export")
    def export() -> Response:
        payload = {"version": 1, "exported_at": time.time(), "reviews": reviews.all()}
        return Response(
            json.dumps(payload, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="habitat-qa-reviews.json"'},
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--renderer-python", type=Path, default=DIMOS_PROJECT_ROOT / "target/habitat/env/bin/python"
    )
    parser.add_argument(
        "--state-dir", type=Path, default=DIMOS_PROJECT_ROOT / "target/habitat/review"
    )
    args = parser.parse_args()
    catalog = build_catalog()
    reviews = FeedbackStore(args.state_dir / "reviews.sqlite3")
    worker = Worker(args.renderer_python, args.state_dir / "renderer.log")
    app = create_app(catalog, reviews, worker)
    uvicorn.run(app, host=args.host, port=args.port, ws_max_size=65536)


if __name__ == "__main__":
    main()
