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

## September 17 desktop performance investigation

The user reported extremely slow simulation and loud laptop fans. Four read-only scene samples from their idle run (`20260917-161002-r1pro-classical-open-space-sim-agent`, seed735730399) measured 0.03046 simulated seconds per wall second. The user stack exited independently; no user process was stopped. GPU rendering used NVIDIA, and available RAM was about 20 GiB with negligible swap use. Do not attribute the problem to software rendering or memory thrashing.

Confirmed scene bottleneck: the hollow cup's 24 wall bottoms were coplanar with its bottom disk, creating 101 simultaneous cup/table contacts. The walls now start at the disk's top while retaining the same top height. The cavity, physical side walls, outer dimensions and total mass remain intact; the small resulting COM/inertia change is physical. In a copied saved slow-run scene, 500 physics steps averaged 2.934 ms before and 0.341 ms after (8.6x improvement), with total contacts 139→43 and cup/table contacts 101→5. Physics timestep and solver settings were not relaxed.

The shared simulator now accepts a positive finite `viewer_fps` independently of sensor `fps`. Other stacks default to 60; both classical blueprints request 15. Expensive viewer synchronizations schedule the next frame from completion. Sensor rendering remains in its existing background thread at 10 Hz; physics retains 2 ms steps. This controls viewer state synchronization, not the native viewer UI's internal redraw rate. Restart the blueprint to regenerate the corrected cup; resetting an existing model will not apply geometry changes.

Validation: 45 focused geometry, open-scene, engine-timing and simulator-module tests passed, including three physical cup variants supported by their base, hollow-cavity rays and physical side-wall rays. Strict mypy passed on four changed production files, with Ruff format/check passing. Native viewer seed5000: trip to display_table passed and the left-hand cup pick passed. Ordinary motion samples reached approximately 1.0x real time; planning phases still slowed physics temporarily. The subsequent requested trip to tall_table failed BEFORE moving at `prepare_carry`: compact carrying posture IK did not converge (first candidate position residual 30.7 mm). Placement was not reached. This is not a passing full-delivery acceptance run; do not conflate the earlier ten-action baseline with this run. All diagnostic workers shut down.

Durable local evidence: `recordings/r1pro-classical-open-space/performance-20260917/validation.json`, raw before/after benchmark, full viewer action reports and snapshots under `fixed-acceptance/`, and test/type-check logs. The full-stack benchmark used single-thread numerical-library limits to bound laptop load. No production thread-limit change was made; interactive launch commands remain unchanged.

Investigation cautions: cProfile on the simulation thread distorted concurrent-viewer performance and its timing results are invalid. Fresh unprofiled viewer/camera and complete-stack checks ran about 0.7x before the cup fix, so the original 0.03x run is not explained conclusively by one cause. A pool-inspection RPC after planning stalled in one diagnostic; do not claim thread counts were proved to be the root cause. No ACT training, new external LLM call, contact disabling or object attachment cheat was introduced.

Remaining follow-up: investigate compact carrying posture selection for the successfully grasped left cup, and reduce planning work that competes with the simulator. Preserve the saved failure and existing grasp/contact gates.

### Viewer preference and object identification

User requested 30 Hz viewer updates after the performance fix. The common classical blueprint now sets `viewer_fps=30` (superseding the initial 15 Hz setting above); physics timestep and sensor rate are unchanged. This setting applies on restart. Documentation clarifies that the generated `cup` is a handleless hollow cylinder. The original measured open-space seed735730399 placed the orange `task_object_5` on `display_table`, with diameter 4.05 cm and height 10.66 cm. The 101-contact measurement came from that saved open-space model, not another scene or a detailed mug asset.


## September 17 follow-up: actual GUI stall and host throttling

The user remained unable to move after the contact optimization. Original run `20260917-165530-r1pro-classical-open-space-sim-agent`, seed493630814, failed empty-hand `go_to(display_table)` in prepare_carry with measured Cartesian endpoint tracking failures. The agent retried after recovery. The scene already had the corrected cup geometry. The original run was stopped gracefully after recovery to instrument the normal CLI stack; HumanCLI was left untouched.

A normal CLI replay (same seed and default OMP4/BLAS2, full native viewer) initially completed the display-table trip at about real time. Later timing captured native viewer `sync()` blocking for 12.42 seconds on the physics thread under the engine lock; simulated time stopped. This is unprofiled timing, not the earlier invalid cProfile experiment.

Implemented optional `background_viewer_rendering`, enabled on the classical blueprints at 30 Hz. The engine publishes small integration-state snapshots to a separate native-viewer process through bounded nonblocking datagrams; full buffers drop frames. The viewer owns its own model/data and does not write physics. Camera requests survive dropped frames. Window exit stops simulation, and shutdown terminates/joins the display process. Linux parent-death signaling avoids orphaned windows. Other blueprints keep their existing inline viewer behavior. Initial thread-only isolation was superseded by process isolation; its full-stack measurements did not establish a performance win.

37 focused engine, IPC and module tests passed: a blocked display cannot stop physics/commands; queued frames cannot backpressure the publisher; newest state/camera requests are displayed; renderer shutdown retains ownership until complete. Native process-viewer startup and cleanup were checked. No successful complete delivery acceptance run has yet been obtained on this version. Do not claim the performance regression is solved.

Crucial host finding: an i9-13980HX measured 400–800 MHz under load, then mostly below 1 GHz even after all diagnostic simulators exited. CPU temperatures were 89–91°C, with package throttling counters actively increasing: 39,730 ms of additional recorded throttling across a 57-second interval spanning shutdown. AC adapter online=1, battery full, powerprofilesctl=performance, CPU maximum configured at 5.4 GHz. Repeated full-stack checks added load; all were stopped to cool the host. No power limits, fan controls, governor, unrelated processes or collision gates were changed. Ask user to check unobstructed cooling before another long test.

Planner traces also captured GIL-holding construction at `build_roboplan_model` -> native Scene. Standalone HomeKinematics initialization ranged from 61.8 seconds to 15.5 seconds, with a later same-process construction taking 9.0 seconds. This variability coincided with severe host throttling; do not label it a proven package regression. Git baseline `eb0a524eb3` is September13; installed RoboPlan0.6.0 binary mtimeSeptember5 and MuJoCo3.10.0 binary mtimeAugust4 predate the successful demo. The classical venv still reads shared main-venv packages through its .pth; this is not a fully isolated dependency lock.

The diagnostic tool environment's LLM credential failed a single agent-send check with401; original user's agent worked. This is NOT evidence that the user's API key is bad. Later checks used local MCP directly, without LLM calls. Restart from user's normal terminal to preserve provider environment.

Evidence is in `/tmp/r1pro-live-performance/`: `live-timings.jsonl` (original inline stall), `navigation-result.json` (initial passing navigation), `v2-stacks.log` equivalent `viewer-stacks.log` (native planner stacks), `v3-live-timings.jsonl` (process-viewer run under throttling), `kinematics-cost.log`, `kinematics-repeat.log`, `thermal-before-stop.json`, `cooldown.jsonl`, `viewer-process-tests.log`. Durable copies are under `recordings/r1pro-classical-open-space/viewer-stall-20260917/`. Diagnostic process viewer was confirmed exited with its parent. Temporary wrappers are not production code.

Next: verify CPU clocks under bounded load after cooling, then repeat the exact user's layout through normal CLI + local MCP: display_table navigation; left object_4 pick; delivery/placement. Compare rates and source hashes; retain honest pending acceptance status until it passes. Also retain the earlier separate seed5000 compact-cup carry IK failure for follow-up.


Correction to the thermal interpretation above: CPU0/first-eight-core readings do not identify the busy simulation core. A bounded saved-state benchmark recorded the actual active core (CPU4) at 1.6–2.5 GHz, with 0.40–0.63 ms per physics step. Thus hardware thermal counters establish pressure but do NOT explain the approximately0.3x live-stack rate on their own. User confirmed vents clear and fans moving air. The earlier 400–800MHz claim about the loaded simulator was too strong and has been corrected in conversation and README. Continue investigating contention; do not block the task solely on cooling. Both full strict mypy on four changed sources and the37 tests passed. A follow-imports=skip experiment produced irrelevant Any/import errors; the normal silent-import check passed.


## Pause for host reboot after thermal slowdown (September 17 desktop session)

User explicitly requested a pause and PC restart. Stop here; resume only when the user returns. Diagnostic simulator and acceptance processes have been stopped. Changes remain uncommitted on `feat/r1pro-classical-apartment` in this worktree; no changes were made to the user's main worktree or host power/fan settings.

Confirmed software finding: inline native viewer synchronization blocked live physics for 12.42 seconds. The new process-isolated snapshot viewer keeps a full desktop window and publishes at 30 Hz without sharing the live physics lock/GIL. A real 12-second SIGSTOP of only the viewer left simulation running at 1.00031x realtime, with scene RPC latency 10–40 ms. Evidence: `recordings/r1pro-classical-open-space/viewer-stall-20260917/process-stall-check.json`. All 37 focused viewer/engine/module tests pass, including after final lifecycle and pacing cleanup. Strict mypy and pre-commit passed before those final small changes; rerun those checks before committing. Native UI may redraw faster than the state update rate; viewer-side physics edits are display-only.

The complete current-seed pick/carry/place acceptance has NOT passed. In v5, navigation to display_table passed (~34 s), but the thin glue stick was pushed over by closure and the two-pad contact gate correctly failed. Recovery rejected the unstable object. After scene reset, worktable navigation passed. The subsequent carton approach lost its simulation worker: kernel reported a segfault at `PyUnstable_InterpreterFrame_GetLine`, and the periodic faulthandler output stopped mid-frame. The temporary `dump_traceback_later` instrumentation is the leading suspect; it is not production code. v6 removed that instrumentation and did not crash before the requested pause, but ran slowly and was stopped during reachability assessment. Do not represent either run as a successful full delivery.

Latest host measurement is stronger than the earlier idle-core readings: over a 3 s interval, worker PID139706/TID140199 used 98.2% of one logical core; its last CPU10 read 199997 kHz. Other active cores read about 200 MHz too. User reports ~90°C even near 2% total CPU, clear vents and fans moving air. Earlier active-core measurements had been 1.6–2.5 GHz with raw MuJoCo at 0.40–0.63 ms/step; those earlier conditions no longer describe the final run. Saved `pause-active-core-frequency.json`. Do not diagnose an exact cooling component from this evidence.

After reboot: first check idle temperature and active-core clock under a brief load, avoiding a long test if severe throttling persists. Then run the full viewer acceptance without periodic native stack dumps or cProfile. Preserve contact, collision and settling checks. Do not change dependency versions without evidence. Repeated HomeKinematics construction takes seconds and is a possible later optimization; no planner cache has been implemented.

Useful temporary scripts and detailed evidence are in `/tmp/r1pro-live-performance/` (may disappear on reboot); copied logs/results live under `recordings/r1pro-classical-open-space/viewer-stall-20260917/`. The normal launch remains `.classical-venv/bin/dimos run r1pro-classical-open-space-sim-agent`, from this worktree. HumanCLI from the same environment. The diagnostic process used a different provider environment, so avoid diagnosing the user's working agent key based on diagnostic authentication errors.
