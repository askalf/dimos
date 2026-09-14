# Open-space classical R1Pro scene

The user requested a larger open scene with 4–5 named platforms at different heights after repeated apartment navigation failures.

Worktree: `/home/mustafa/dimos-wt/r1pro-classical-apartment`.
Branch: `feat/r1pro-classical-apartment`.
Previous pushed baseline: `b9367026a6`.

## Implementation

A 12 × 12 m bounded floor with five colored, labeled physical platforms: `low_bench` 60 cm, `worktable` 70 cm, `display_table` 80 cm, `high_counter` 85 cm and `tall_table` 90 cm. Platform centers are separated by more than 3 m. The original tray remains on the worktable. The five existing procedural props are randomized across the platforms, one per support. No apartment scene package is needed.

New blueprints: `r1pro-classical-open-space-sim` and `r1pro-classical-open-space-sim-agent`. Full MuJoCo viewer and Zenoh defaults; the agent starts idle and supports HumanCLI. Existing GraspGenX and DimOS independent pick, hold, navigate, place and recovery skills are reused. No ACT runs. The apartment blueprint retains its original scene.

Scene creation is a protected lifecycle hook in `primitive_sim.py`; the open simulator subclass supplies its generator and platform navigation headings. The common classical blueprint factory accepts the simulator class. The registry was regenerated using its test.

Instructions: `dimos/robot/galaxea/r1pro/OPEN_SPACE.md`.

## Validation and fixes

- Real robot scene compiled; all five initial objects upright and physically supported; finite navigation cloud; labels and full-scene framing inspected. Final 85 cm counter rendering: `/tmp/r1pro-open-space/overview-final.png`.
- 68 focused tests, six registry checks and strict mypy on ten changed production sources passed. Both blueprint names appear in CLI listing.
- Native v1 completed pick, low-bench navigation/placement/re-pick, then stopped after 41 mm overshoot at the high-counter route endpoint. Transit now stops at its existing 25 mm arrival tolerance, verifies the stopped footprint, and retains the 40 mm deviation guard. A focused regression covers early cancellation.
- Native v2 passed that navigation but failed a 95 cm preplace transfer despite feasible endpoint IK. Removed experiments with stance ordering, retraction and relaxed intermediate IK. The retained planner change checks a complete upright transfer on a snapshot before accepting a placement candidate. Counter height is now 85 cm; the tallest table remains 90 cm.
- Native v3 physically placed and released the carton upright on the 85 cm counter, then failed retreat because arm-only IK jumped branches. The planner previously tried torso assistance only on nonconvergence, not discontinuity. It now tries the same torso-assisted solve for either case, preserves the other hand, and applies the same joint-step and collision gates. Saved failure replay now returns 27 valid retreat waypoints (`retreat-replay-fixed.log`); no live object teleporting or softened collision tests.

- Native v4 passed all navigation and the high-counter release/retreat. The left cup pick stopped before physical motion: an inactive right wrist measurement was 3.8 microradians outside the SDK's conservative margin. `HomeKinematics.solve` had seeded the SDK with bounded measurements, then overwritten untouched joints with raw measurements. It now retains the same bounded measured seed for unsolved joints. Active solved joint limits are still strictly validated. Saved-state reassessment and staging pass (`pick-replay-fixed.log`).

## Completed acceptance run

Native v5 passed all ten actions at seed5000 in one continuous run: right carton pick; low-bench navigation, placement and re-pick; high-counter navigation and placement; display-table navigation; left cup pick; tall-table navigation and placement. The carton and cup ended physically supported, released, upright and settled on their requested surfaces. No recovery or scene reset occurred. Source hashes were checked against the current implementation after the run.

A second full GLFW desktop startup at seed5001 passed, with one supported object per platform. Both native and viewer stacks shut down all workers cleanly; no validation job remains running. No external LLM call was made. The agent prompt and HumanCLI path reuse the existing classical stack, with the new exact platform names.

Durable evidence: `recordings/r1pro-classical-open-space/seed-5000/` (gitignored):

- `validation.json`: action list, held-object transitions and tested source hashes.
- `native-v5/result.json` and `native-v5/session/action-*.json`: full outcomes and physical states.
- `desktop-startup/result.json` and `desktop.log`: full-viewer startup and clean shutdown.
- `overview-final.png`: final five-platform layout.
- `tests-v5.log`, `mypy-v5.log`, `registry-final.log`: automated checks.
- `retreat-replay-fixed.log` and `pick-replay-fixed.log`: saved failure reproductions and corrected outcomes.

Earlier failed runs remain in `/tmp/r1pro-open-space/native-v1` through `native-v4` for diagnosis. All simulation checks used private MCP10026 and Zenoh224.0.0.224:19492 with bounded output and disk-space guards. No user stack was stopped.

## Launch

```bash
cd /home/mustafa/dimos-wt/r1pro-classical-apartment
.classical-venv/bin/dimos run r1pro-classical-open-space-sim-agent
```

In another terminal, from the same directory: `.classical-venv/bin/dimos humancli`.
The demo starts idle, uses the full viewer and Zenoh, and randomizes the layout seed on each launch. `reset_scene` repeats the current seed. See `OPEN_SPACE.md` for exact support names and example requests.
