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

"""``dimos map live`` -- ask a recording the way the robot would, and watch it happen.

    dimos map live bike.db -q "a traffic cone" -q "a stop sign" -o ~/out

Nothing here is a simulation of the live system: it builds the same `LiveQuery` the
`Hyperspace` module holds, warms the same models, and calls the same `ask`. The only
difference is where the store came from. That is the point -- a page drawn by a
reimplementation would be a picture of something the robot does not do.

Each query gets a self-contained HTML page that replays the answering: the clock runs
from the moment the question was asked, and each answer appears at the second it
actually arrived, with the frame the detector looked at and the box it drew on it. The
pages, an index and a `queries.zip` of all of them land in the output directory.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any
import zipfile

import numpy as np
import typer

from dimos.mapping.hyperspace.cli import open_store, pick_device, pick_stream
from dimos.mapping.hyperspace.detect import DetectConfig
from dimos.mapping.hyperspace.frames import member_streams, spec_of
from dimos.mapping.hyperspace.live import LiveConfig, LiveQuery
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from dimos.mapping.hyperspace.detect import Detection

logger = setup_logger()


def slug_of(text: str) -> str:
    """A filename from a query: ``"a stop sign"`` -> ``a_stop_sign``."""
    kept = [character if character.isalnum() else "_" for character in text.strip().lower()]
    return "_".join(part for part in "".join(kept).split("_") if part) or "query"


def _picture(answer: Detection, width: int = 720) -> str:
    """The frame the detector looked at, with its box drawn on, as a data URI."""
    from PIL import Image as PILImage

    from dimos.mapping.hyperspace.render import draw_box

    if answer.image is None:
        return ""
    # `to_rgb()` already returns RGB. Reversing the channels here on the assumption it
    # returns BGR is how the pages came out with blue traffic cones -- the same
    # confusion that put the swap into the stored frames in the first place.
    picture = PILImage.fromarray(np.asarray(answer.image.to_rgb().data)).convert("RGB")
    if answer.box2d is not None:
        draw_box(picture, answer.box2d, (255, 138, 46), width=max(2, picture.width // 180))
    if picture.width > width:
        picture = picture.resize((width, round(picture.height * width / picture.width)))
    buffer = io.BytesIO()
    picture.save(buffer, format="JPEG", quality=72, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def page(query: str, result: Any, answers: list[Any], recording: str, loaded: dict) -> str:
    """One query's page: the answers, in the order and at the times they arrived."""
    rows = []
    for answer in answers:
        if answer.box3d is None:
            rows.append(
                {
                    "arrived": round(float(answer.arrived), 3),
                    "found": False,
                    "note": answer.note or "nothing found",
                    "frames": answer.episode_frames,
                }
            )
            continue
        box = answer.refined or answer.box3d
        rows.append(
            {
                "arrived": round(float(answer.arrived), 3),
                "found": True,
                "score": round(float(answer.score), 3),
                "centre": [round(float(v), 2) for v in box.centre],
                "extent": [round(float(v), 2) for v in box.extent],
                "depth": round(float(box.depth_m), 2),
                "frame": box.frame,
                "camera_frame": answer.camera_frame,
                "stamp": round(float(answer.ts), 3),
                "place": answer.place_id,
                "duplicate_of": answer.duplicate_of,
                "frames": answer.episode_frames,
                "models": answer.models,
                "image": _picture(answer),
            }
        )
    timings = {name: value for name, value in result.timings.items() if name != "frames_matched"}
    payload = {
        "query": query,
        "recording": recording,
        "answers": rows,
        "ms": result.ms,
        "timings": timings,
        "frames_matched": int(result.timings.get("frames_matched", 0)),
        "refused": result.refused,
        "places": len(result),
        "loaded": {name: round(value, 2) for name, value in loaded.items()},
    }
    return _PAGE.replace("__DATA__", json.dumps(payload, separators=(",", ":")))


def main(
    recording_path: Path = typer.Argument(..., help="the .db recording to ask"),
    query: list[str] = typer.Option(..., "-q", "--query", help="what to look for; repeatable"),
    out: Path = typer.Option(None, "--out", "-o", help="where the pages and the zip land"),
    models: str = typer.Option(
        "", "--models", help="comma separated member tags to search (default: all of them)"
    ),
    threshold: float = typer.Option(
        DetectConfig.threshold, "--threshold", help="OWLv2's per-box acceptance score"
    ),
    max_episodes: int = typer.Option(12, "--max-episodes"),
    merge_m: float = typer.Option(0.75, "--merge", help="answers this close are one place (m)"),
    world_frame: str = typer.Option("odom", "--world-frame"),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Answer each query the way the live module would, and write a page per query."""
    recording_path = recording_path.expanduser()
    out = (out or recording_path.parent / f"{recording_path.stem}_live").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    store = open_store(recording_path)

    available = [tag for tag, _ in member_streams(store)]
    if not available:
        raise typer.BadParameter(
            f"{recording_path} holds no patch streams; build one with `dimos map embed`"
        )
    wanted = [tag.strip() for tag in models.split(",") if tag.strip()] or available
    missing = [tag for tag in wanted if tag not in available]
    if missing:
        raise typer.BadParameter(f"no such model(s) {missing}; the index holds {available}")

    live = LiveQuery(
        store,
        LiveConfig(
            detect=DetectConfig(
                threshold=threshold,
                device=pick_device(device),
                max_episodes=max_episodes,
                world_frame=world_frame,
            ),
            models=wanted,
            merge_m=merge_m,
            # Named at construction, not patched afterwards: `RecordingFrames` reads the
            # camera intrinsics in its constructor, so a name set later is set too late.
            color_stream=pick_stream(store, None, "color", "image"),
            depth_stream=pick_stream(store, None, "depth", "image"),
            color_info_stream=pick_stream(store, None, "camera", "info"),
            depth_info_stream=pick_stream(store, None, "depth", "camera", "info"),
        ),
    )

    typer.echo(f"index: {recording_path}  models {wanted} of {available}")
    loaded = live.warm([spec_of(tag) for tag in wanted])
    typer.echo(
        f"warm: detector {loaded['detector']:.1f}s, text towers {loaded['towers']:.1f}s, "
        f"recording {loaded['recording']:.1f}s, {int(loaded['index'])} patches in "
        f"{loaded['index_s']:.1f}s"
    )

    written = []
    summary: dict[str, Any] = {"recording": str(recording_path), "queries": {}}
    for text in query:
        typer.echo(f"\n{text!r}")
        started = time.monotonic()
        result = live.ask(text)
        for found in result.objects:
            typer.echo(
                f"  place {found.place_id:<3} owl {found.confidence:.2f}  "
                f"at ({found.centre[0]:6.1f}, {found.centre[1]:6.1f}, {found.centre[2]:5.1f}) "
                f"{found.frame}  {found.depth_m:.1f} m away  {found.views} view(s)"
            )
        typer.echo(
            f"  {len(result)} place(s), {result.refused} refused, "
            f"{time.monotonic() - started:.1f}s  {result.timings}"
        )
        path = out / f"{slug_of(text)}.html"
        path.write_text(page(text, result, live.answers, recording_path.name, loaded))
        written.append(path)
        typer.echo(f"  {path}")
        summary["queries"][text] = result.as_dict()

    index = out / "index.html"
    index.write_text(_index_page(recording_path.name, query, written, summary))
    (out / "answers.json").write_text(json.dumps(summary, indent=2, default=str))

    bundle = out / "queries.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in [index, *written]:
            archive.write(path, path.name)
    typer.echo(f"\n{len(written)} page(s) + {index.name} -> {bundle}")
    live.close()


def _index_page(recording: str, queries: list[str], written: list[Path], summary: dict) -> str:
    items = []
    for text, path in zip(queries, written, strict=False):
        answer = summary["queries"][text]
        items.append(
            f"<li><a href='{path.name}'>{text}</a> &mdash; {len(answer['objects'])} place(s), "
            f"{answer['refused']} refused, {answer['ms']:.0f} ms</li>"
        )
    return (
        "<!doctype html><meta charset='utf-8'><title>hyperspace, live</title>"
        "<style>body{font:15px/1.6 ui-sans-serif,-apple-system,sans-serif;max-width:60ch;"
        "margin:40px auto;padding:0 20px;background:#0a0b10;color:#e8e8ef}"
        "a{color:#ff8a2e}li{margin:6px 0}</style>"
        f"<h1>{recording}</h1><ul>{''.join(items)}</ul>"
    )


_PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>hyperspace, live</title>
<style>
:root { --ink:#e8e8ef; --dim:#9a9aa8; --edge:#2a2b36; --panel:#14151c; --hot:#ff8a2e;
        color-scheme: dark }
* { box-sizing:border-box }
body { margin:0; background:#0a0b10; color:var(--ink); padding:28px 20px 80px;
       font:14px/1.6 ui-sans-serif,-apple-system,"Segoe UI",sans-serif }
main { max-width:1100px; margin:0 auto }
h1 { font-size:24px; margin:0 0 2px }
h1 em { color:var(--hot); font-style:normal }
.sub { color:var(--dim); margin:0 0 20px }
#bar { position:sticky; top:0; background:#0a0b10ee; backdrop-filter:blur(6px);
       padding:12px 0 14px; z-index:5; border-bottom:1px solid var(--edge); margin-bottom:18px }
#clock { color:var(--hot); font-variant-numeric:tabular-nums; font-size:22px; font-weight:600 }
#track { height:4px; background:var(--edge); border-radius:2px; margin-top:8px; overflow:hidden }
#track i { display:block; height:100%; width:0; background:var(--hot) }
button { background:var(--panel); color:var(--ink); border:1px solid var(--edge);
         border-radius:6px; padding:6px 14px; cursor:pointer; font:inherit }
.split { display:flex; gap:14px; align-items:baseline; flex-wrap:wrap }
.answer { border:1px solid var(--edge); border-radius:10px; background:var(--panel);
          margin-bottom:14px; overflow:hidden; opacity:0; transform:translateY(8px);
          transition:opacity .35s, transform .35s }
.answer.here { opacity:1; transform:none }
.answer .head { display:flex; gap:14px; align-items:baseline; padding:12px 16px;
                border-bottom:1px solid var(--edge); flex-wrap:wrap }
.at { color:var(--hot); font-variant-numeric:tabular-nums; font-weight:600; min-width:70px }
.score { font-weight:600 }
.where { color:var(--dim) }
.answer img { display:block; width:100%; }
.refused .head { color:var(--dim) }
.meta { color:var(--dim); font-size:13px; padding:10px 16px }
</style>
<main>
<h1>&ldquo;<em id="query"></em>&rdquo;</h1>
<p class="sub" id="sub"></p>
<div id="bar">
  <div class="split"><button id="replay">replay</button><span id="clock">0.00s</span>
  <span class="where" id="live"></span></div>
  <div id="track"><i></i></div>
</div>
<div id="answers"></div>
<p class="meta" id="loaded"></p>
</main>
<script>
const data = __DATA__

document.getElementById("query").textContent = data.query
const split = Object.entries(data.timings).map(([k, v]) => `${k} ${v.toFixed(2)}s`).join(" + ")
document.getElementById("sub").textContent =
    `${data.recording} \\u00b7 ${data.places} place(s), ${data.refused} refused \\u00b7 `
    + `${(data.ms / 1000).toFixed(2)}s total \\u00b7 ${split}`
    + ` \\u00b7 ${data.frames_matched} frames matched`
document.getElementById("loaded").textContent =
    "loaded before the question: " + Object.entries(data.loaded)
        .map(([k, v]) => `${k} ${v}`).join(", ")

// Every answer carries the second it arrived, measured from the question, so the page
// can play the query back rather than presenting the result as if it were instant.
const holder = document.getElementById("answers")
const cards = data.answers.map(answer => {
    const card = document.createElement("div")
    card.className = "answer" + (answer.found ? "" : " refused")
    const head = document.createElement("div")
    head.className = "head"
    const at = document.createElement("span")
    at.className = "at"
    at.textContent = answer.arrived.toFixed(2) + "s"
    head.appendChild(at)
    if (answer.found) {
        const score = document.createElement("span")
        score.className = "score"
        score.textContent = "owl " + answer.score.toFixed(2)
        const where = document.createElement("span")
        where.className = "where"
        where.textContent = `(${answer.centre.join(", ")}) ${answer.frame}`
            + ` \\u00b7 ${answer.extent.join(" x ")} m \\u00b7 ${answer.depth} m away`
            + ` \\u00b7 ${answer.camera_frame} @ ${answer.stamp}`
            + (answer.duplicate_of ? ` \\u00b7 another look at #${answer.duplicate_of}` : "")
        head.append(score, where)
    } else {
        const note = document.createElement("span")
        note.className = "where"
        note.textContent = answer.note + ` \\u00b7 ${answer.frames} frame(s)`
        head.appendChild(note)
    }
    card.appendChild(head)
    if (answer.image) {
        const img = document.createElement("img")
        img.src = answer.image
        img.loading = "lazy"
        card.appendChild(img)
    }
    holder.appendChild(card)
    return card
})

const last = Math.max(0.001, ...data.answers.map(a => a.arrived))
const clock = document.getElementById("clock")
const progress = document.querySelector("#track i")
const live = document.getElementById("live")
let from = performance.now()

function showUpTo(seconds) {
    cards.forEach((card, i) => card.classList.toggle("here", seconds >= data.answers[i].arrived))
    const shown = data.answers.filter(a => seconds >= a.arrived).length
    clock.textContent = Math.min(seconds, last).toFixed(2) + "s"
    progress.style.width = Math.min(100, (seconds / last) * 100) + "%"
    live.textContent = shown ? `${shown} of ${data.answers.length} in` : "searching\\u2026"
}

// A timer, not requestAnimationFrame: a tab that is not visible gets no frames, and a
// page whose answers never appear is worse than one that animates coarsely. Thirty a
// second is finer than anyone reads a clock.
function tick() {
    showUpTo((performance.now() - from) / 1000)
}
document.getElementById("replay").onclick = () => { from = performance.now(); tick() }
setInterval(tick, 33)
tick()
</script>
"""


if __name__ == "__main__":
    typer.run(main)
