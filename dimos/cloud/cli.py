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

"""CLI-facing operations for `dimos data`; commands delegate here one-line each."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import contextlib
from datetime import datetime, timezone
import functools
from pathlib import Path
import shutil
from typing import Any

import typer

from dimos.cloud.data import CloudData, recordings


def tz_label() -> str:
    """The zone the table header advertises, e.g. PDT."""
    return datetime.now().astimezone().tzname() or ""


def local_time(ts: str, label: str | None = None) -> str:
    """ISO timestamp (UTC when naive) -> local wall time; a row whose zone differs
    from `label` (a DST boundary) carries its own."""
    try:
        d = datetime.fromisoformat(ts)
    except ValueError:
        return ts[:16].replace("T", " ")
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    d = d.astimezone()
    label = tz_label() if label is None else label
    suffix = "" if d.tzname() == label else f" {d.tzname()}"
    return d.strftime("%Y-%m-%d %H:%M") + suffix


def handle_fail(fn: Callable[..., None]) -> Callable[..., None]:
    @functools.wraps(fn)
    def wrapper(*a: Any, **kw: Any) -> None:
        try:
            fn(*a, **kw)
        except (RuntimeError, OSError) as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1) from e

    return wrapper


# What the bar says for each phase the backend reports. Progressive verbs: the
# bar is a status line, and a bare "compress" read as an instruction.
_PHASE = {
    "compress": "compressing",
    "checksum": "hashing",
    "upload": "uploading",
    "download": "downloading",
    "verify": "verifying",
    "decompress": "decompressing",
}


def _squeeze(r: dict[str, Any]) -> str:
    """' · compressed 2.8 GB → 1.1 GB (61% smaller)' when it was, else ''.

    The uploaded bytes are the compressed size, which does not match the file on
    disk; this is the line that explains the gap once the transient bar is gone.
    """
    from rich.filesize import decimal

    enc, raw, wire = r.get("content_encoding"), r.get("raw_bytes"), r.get("wire_bytes")
    if not (enc and raw and wire):
        return ""
    pct = int((1 - wire / raw) * 100)  # floor, so a 99.6% ratio never reads "100% smaller"
    gain = (
        f" ({pct}% smaller)" if pct > 0 else ""
    )  # already-compact data can grow; don't claim a gain
    return f" · compressed {decimal(raw)} → {decimal(wire)}{gain}"


def _short(name: str, n: int = 30) -> str:
    """Middle-elide a long filename so the progress line fits a narrow terminal,
    keeping both ends — a recording is told apart by its date suffix."""
    if len(name) <= n:
        return name
    keep = n - 1
    return name[: keep - keep // 2] + "…" + name[len(name) - keep // 2 :]


_WIDE = 90  # columns; below this the bar sheds speed and ETA to stay one row


def _progress_columns(width: int) -> list[Any]:
    """Columns for the upload/pull bar, shed to fit the terminal.

    The bar flexes to the space left over (``bar_width=None``), and the speed and
    ETA drop on a narrow terminal so the line stays one row instead of wrapping —
    a wrapped bar is what smears when the window is resized mid-transfer.
    """
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    cols: list[Any] = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        DownloadColumn(),
    ]
    if width >= _WIDE:
        cols += [TransferSpeedColumn(), TimeRemainingColumn()]
    return cols


class _Ticker:
    """Feeds backend progress into one rich task per phase.

    A fresh task per phase gives an indeterminate phase a pulsing bar instead of
    "0%" of the previous phase's total (rich reads total=None as "unchanged"),
    and a speed and ETA that describe this phase rather than the last one.
    """

    def __init__(self, bar: Any, name: str, width: int | None = None) -> None:
        self.bar, self.name, self.phase = bar, _short(name), "reading"
        self.wide = None if width is None else width >= _WIDE
        self.task = bar.add_task(f"reading {self.name}", total=None)

    def __call__(self, phase: str, done: int, total: int) -> None:
        if self.wide is not None:  # shed or restore speed and ETA as the window changes
            cols = shutil.get_terminal_size((100, 24)).columns
            if (cols >= _WIDE) != self.wide:
                self.bar.columns = tuple(_progress_columns(cols))
                self.wide = cols >= _WIDE
        if phase != self.phase:
            self.bar.remove_task(self.task)
            label = f"{_PHASE.get(phase, phase)} {self.name}"
            self.task = self.bar.add_task(label, total=total or None)
            self.phase = phase
        self.bar.update(self.task, completed=done, total=total or None)


@contextlib.contextmanager
def _bar(name: str) -> Iterator[Callable[[str, int, int], None]]:
    """A transfer bar that survives a window resize.

    Drawn on the alternate screen and repainted whole every frame, the same fix
    as the login wait. An inline bar cannot be made safe: once the terminal
    reflows a wide line into two rows, the cursor-up redraw lands under the old
    frame instead of over it. The bar is transient either way; the line that
    stays is the summary printed after it.
    """
    from dimos.cli import theme

    if not theme.enabled():  # piped or CI: no bar to draw, just the line printed after
        yield lambda phase, done, total: None
        return

    from rich.live import Live
    from rich.progress import Progress

    width = shutil.get_terminal_size((100, 24)).columns
    progress = Progress(*_progress_columns(width), auto_refresh=False)  # the Live below renders it
    with theme.muted_input(), Live(progress, screen=True, transient=True, refresh_per_second=10):
        yield _Ticker(progress, name, width)


@handle_fail
def upload(
    path: Path | None, robot: str | None, kind: str | None, since_s: float | None, chunk: int | None
) -> None:
    explicit = path is not None
    path = None if str(path) == "latest" else path
    cloud = CloudData()
    targets = recordings(since_s) if since_s else [path] if path else recordings()[-1:]
    if not targets:
        raise RuntimeError("nothing to upload — pass a path")
    failed = False
    for t in targets:
        try:
            with _bar(t.name) as tick:
                r = cloud.upload(
                    t,
                    robot_id=robot,
                    kind=kind,
                    chunk_mb=chunk,
                    progress=tick,
                    skip_recent=not explicit,
                )
            note = "already uploaded" if r["skipped"] else r["state"]
            typer.echo(f"{t.name}: {note}{_squeeze(r)} ({r['upload_id'][:12]})")
            # An already-complete upload comes back from create, which carries no
            # quota; only a fresh completion does. Missing means nothing to warn.
            quota = r.get("quota") or {}
            if quota.get("state") not in (None, "ok"):
                typer.echo(quota["message"], err=True)
        except (RuntimeError, OSError) as e:
            typer.echo(f"{t.name}: {e}", err=True)
            failed = True
    if failed:
        raise typer.Exit(1)


@handle_fail
def ls() -> None:
    import sys

    if sys.stdout.isatty() and sys.stdin.isatty():  # Textual needs a real TTY
        from dimos.cloud.tui import DataBrowser

        DataBrowser().run()
        return

    from rich import box
    from rich.console import Console
    from rich.filesize import decimal
    from rich.table import Table

    tz_now = tz_label()
    rows = CloudData().ls()
    org = any(u.get("uploader_email") for u in rows)
    table = Table(box=box.SIMPLE_HEAVY, header_style="bold")
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("file", style="bold")
    table.add_column(f"uploaded ({tz_now})", style="dim", no_wrap=True)
    table.add_column("kind")
    if org:
        table.add_column("uploader", style="dim")
    table.add_column("blueprint", style="magenta")
    table.add_column("robot", style="magenta")
    table.add_column("topics", style="dim", max_width=48)
    table.add_column("size", justify="right")
    table.add_column("state")
    for u in rows:
        mani = u.get("manifest") or {}
        state = u["state"]
        table.add_row(
            u["id"][:12],
            u["filename"],
            local_time(str(u.get("created_at") or ""), tz_now) or "—",
            u.get("kind", ""),
            *([u.get("uploader_email") or "—"] if org else []),
            mani.get("blueprint") or "—",
            u.get("robot_id") or "—",
            ", ".join(s.get("name", "?") for s in (mani.get("streams") or [])) or "—",
            decimal(u["size"]),
            f"[green]{state}[/]" if state == "complete" else f"[yellow]{state}[/]",
        )
    Console().print(table)


@handle_fail
def pull(upload_id: str | None, dest: Path | None) -> None:
    upload_id = None if upload_id == "latest" else upload_id
    cloud = CloudData()
    row = cloud.resolve(upload_id)
    with _bar(row["filename"]) as tick:
        out = cloud.pull(str(row["id"]), dest, progress=tick)
    typer.echo(f"pulled to {out}")


@handle_fail
def status(upload_id: str) -> None:
    s = CloudData().status(upload_id)
    typer.echo(s["state"] + (f" — parts on server: {len(s['parts'])}" if s["parts"] else ""))


@handle_fail
def quota() -> None:
    from rich.filesize import decimal

    q = CloudData().quota()
    lim = q["limits"]
    typer.echo(
        f"{q['pct']}% used ({q['state']}) — total {decimal(q['used_total'])}"
        f" of {lim['total_gb']} GB, today {decimal(q['used_today'])}"
        f" of {lim['daily_gb']} GB"
    )
