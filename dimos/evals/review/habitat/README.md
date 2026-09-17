# Habitat QA review dashboard

Browse the 18 scene suites and 210 user-reviewed questions with a live Habitat
top-down map and optional 3D inspection. Approve, flag, or mark questions for removal; propose wording, options,
and answer corrections; add questions and comments. Feedback is separate from
the executable suites and autosaves to SQLite.

## Launch

From the QA checkout, with its own `.venv` and installed Habitat assets:

```bash
.venv/bin/python -m dimos.evals.review.habitat.server \
  --host "$(tailscale ip -4)" --port 8876
```

Open `http://<this-machine-tailscale-ip>:8876` on another device in the same
tailnet. The selected port must be allowed by your tailnet policy. Binding to
the Tailscale address makes the service available on that interface. This is a
trusted authoring tool: anyone with access to that endpoint can control the
shared camera and edit reviews. No public relay or external frontend assets are
required. Drag-to-look works over HTTP without pointer-lock/HTTPS requirements.

Optional arguments:

- `--renderer-python`: defaults to `target/habitat/env/bin/python` (Habitat-Sim
  0.3.3 / Python 3.9). The worker uses its own process and a private socket.
- `--state-dir`: defaults to `target/habitat/review`; contains `reviews.sqlite3`
  and `renderer.log`. Preserve this directory to retain reviews.
- `HABITAT_REVIEW_ASSET_ROOT`: fallback dataset root, defaulting to
  `~/Documents/habitat-sim/data`. Existing suite dataset environment overrides
  also work. HM3D defaults to the project-local downloaded example dataset.

## Review

Select a scene to see its whole footprint from directly above:

- Drag to pan; scroll or use the +/- buttons to zoom.
- **Fit entire map**, or double-click, restores the whole scene footprint.
- Every scene loads fully fitted, using full geometry bounds with an 8% border.
  Canvas resizing fits the entire source image without cropping.
- **Floor** selects a detected level in multi-storey scenes.
- **Room slice** clips geometry above the selected floor (default 1.8 m) so
  ceilings and roofs do not hide the furniture. Objects above that cut are hidden.
- In 2D, pan and zoom are instantaneous browser-side transforms of the
  2048×1536 orthographic map.

Switch to **3D Inspect** to tilt and turn the real scene:

- Drag to orbit and tilt; scroll or use +/- to move closer/farther.
- Shift-drag or right-drag pans across the scene.
- **Reset 3D view** fits the whole scene again.
- **Ceiling cutaway** removes geometry above the selected room slice with a
  horizontal clipping plane that stays level as you orbit. Turn it off to show
  the full structure, including roofs. With cutaway enabled, the camera stays
  above the cut plane.
- **2D Map** returns to the previous map pan/zoom. Every newly loaded scene
  starts in fully fitted 2D mode.

Mode switching reuses the loaded simulator and does not reload the dataset.
3D interactions request new RGB frames; input is coalesced to one in-flight
render so mouse events do not accumulate into a delayed command queue.

Question references are extracted from the actual suite scorers and checked for
full-credit agreement during startup. Unsupported scorer forms fail explicitly.
The selected scene's section of `dimos/evals/suites/habitat/SCENES.md` is shown
under **Scene context & dataset notes**. This document describes the environments;
questions and reference answers are maintained in the Python suites.

Comments and proposed edits save after a short debounce. **Saved ✓** confirms the
server write. Unsent changes are backed up in that browser's local storage;
**Retry saves** resubmits them. Concurrent stale edits get an explicit conflict
rather than overwriting another tab's saved review. **Export feedback** downloads
JSON including original question/reference snapshots and the camera position at
the time of the edit. A reset to "unreviewed" changes the status, not the notes.

Applied validation suggestions are displayed under their new suite IDs, with
their original notes available in a collapsed section. Retired questions are
hidden from the current question list. SQLite and JSON export retain the entire
history; a later review under a canonical ID takes precedence over its old alias.

One renderer is shared across browser tabs; load one scene at a time. HSSD and
ReplicaCAD navmeshes are rebuilt in memory to estimate floor elevations. Map
framing uses all scene geometry, including areas outside the navmesh. Detected
floor levels are approximate; adjusting the room slice can expose scan details.
ReplicaCAD articulation is loaded with physics enabled, but physics is never
stepped. Dataset files and navmeshes on disk are not changed.

## Checks

```bash
.venv/bin/pytest dimos/evals/review/habitat/test_review.py -q
```

These checks validate catalog/reference alignment, persistent feedback,
revision conflicts, adding questions, export, and camera API validation without
requiring a GPU. Live scene and browser validation should use a separate
`--state-dir` so smoke-test comments do not enter the real review database.
