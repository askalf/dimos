# Five-bottle ACT demo: portable snapshot

Branch: **feat/r1pro-act-sim**. This folder ships the complete house scene and the
pinned R1Pro robot meshes/URDF. The actual trained ACT model, preprocessing and
normalization files are committed at
`recordings/r1pro-act-task/policy-packing-augmented`.
These demo assets are ordinary Git blobs; they need no separate LFS download.

## On the other Linux laptop

Use a desktop session with working NVIDIA/CUDA and OpenGL, Python 3.12, uv and
Rust/Cargo. Python/native dependencies must be installed for that machine;
virtual environments and credentials are deliberately not copied.

```bash
git clone --branch feat/r1pro-act-sim --single-branch git@github.com:dimensionalOS/dimos.git
cd dimos
uv sync --extra all --extra learning
uv run python demo_assets/r1pro-five-bottle/restore.py
cargo build --release --locked -p dimos-mls-planner --bin mls_planner
uv run dimos run r1pro-home-sim
```

The isolated LeRobot runtime installs its pinned dependencies on first use.
The demo opens the full MuJoCo window, packs five bottles with ACT, picks up the
tray with both hands using classical planning, navigates through the apartment,
and places it at the laptop table. The automatic demo needs no language-model key.

For the interactive bottle/tray version, use `uv run dimos run r1pro-home-sim-agent`
and `uv run dimos humancli` in a second terminal. Configure your own provider key
in the launching terminal. This is the tested narrow bottle task, not the later
experimental random-object or independent-primitive policy.

`restore.py` verifies every packaged asset/checkpoint checksum, restores the house
to the expected location, and seeds a local robot-description cache when absent.
Existing scene/cache directories are preserved. `--check-only` just verifies files.
The robot cache intentionally contains an untracked vendor snapshot so the normal
asset loader uses it without replacing it from upstream.

`validated-home-5000.json` records the successful full-window Zenoh delivery.
`INVENTORY.md` and `inventory.json` preserve the broader historical project handoff;
their original local paths and branch-ahead counts describe the earlier audit.
Model weights and processors are byte-identical to the validated deployment.
The fresh laptop install has not been physically revalidated during this upload.

Detailed behavior: `dimos/robot/galaxea/r1pro/BOTTLE_PACKING.md`.
