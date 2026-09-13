# Classical apartment handoff — 2026-09-13

The user switched from ACT to classical manipulation with GraspGen and requested a separate branch. No ACT training or policy executes in this demo.

Branch: `feat/r1pro-classical-apartment`. Permanent checkout: `/home/mustafa/dimos-wt/r1pro-classical-apartment`. The former `/tmp/dimos-r1pro-primitives` path is a compatibility symlink. The existing ACT branch/checkpoints and Alfred checkout were not modified. The isolated runtime's launchers and package path were relocated with the worktree.

## Current verified state

Main was fetched again before publication. The only new upstream change was RealSense Nix build fix `aa8a158469`. Recreating historical merges during rebase reintroduced old conflicts, so that rebase was aborted and upstream was merged as `28eeed4550`. Comparing against the verified pre-merge tree changes only `dimos/hardware/sensors/camera/realsense/rust/flake.nix`; all demo sources are identical. The branch contains current main.

- **Complete native delivery passed:** `/tmp/r1pro-classical-bootstrap/delivery-v19/result.json`, six successful actions: right pick of object_2, dining navigation, dining placement, re-pick from its new position, kitchen navigation, kitchen placement. The bottle was upright and held during both trips, then upright, settled, released and supported by the correct furniture after each placement. Both placements recorded force-backed intended support before opening; kitchen needed 0.70 mm additional descent. Source checkpoint: `f351049fac`.
- **Other successful native scenarios:** `bimanual-v13` picked with both hands then placed independently while retaining the other hold; `carton-v12` navigated to the kitchen and picked/placed a carton; `cup-v12` used the left hand, turned the base locally, and placed in the tray. These earlier scenarios predate the numerical-limit and placement-budget changes; the full six-action run uses both final fixes.
- **Desktop startup passed after relocation:** `/tmp/r1pro-classical-bootstrap/relocated-startup/result.json` and `promotion-status.json`. Real GLFW MuJoCo window, all native modules, GraspGenX, five objects and local MCP inventory; clean shutdown. No external model request.
- **Launch:** follow `dimos/robot/galaxea/r1pro/CLASSICAL_APARTMENT.md`. Use `.classical-venv/bin/dimos run r1pro-classical-apartment-sim-agent`, then `.classical-venv/bin/dimos humancli` in a second terminal from this checkout. Zenoh and full viewer are defaults; no shell exports or environment activation required.
- **Validation:** 73 earlier focused tests; subsequent execution/navigation/state checks passed as recorded below. Final numerical-limit regression: 5 passed, strict mypy passed both affected sources, and all commit hooks passed. Do not sum overlapping test batches into a unique test count.
- **No background work remains:** delivery and promotion supervisors completed; test stacks shut down. No training is running. Other user processes were not stopped by this work.
- **Pending:** the external language-provider/HumanCLI round trip was not run. Automatic approval review rejected sending simulated inventory/tool context to the configured provider, and the user-facing approval question remains unanswered. Local MCP physical actions are verified. Do not send a prompt until that approval arrives.

## Scope and remaining limits

The pipeline strictly resolves object descriptions, raycasts segmented object geometry, proposes real GraspGenX grasps, ranks body/arm reachability, stages through DimOS IK/planning, and executes independent pick/hold/place commands. KronkNav uses the apartment cloud and the DimOS holonomic controller executes paths after full robot/cargo collision checks. Live holding and support use contact physics; attachments exist only in planning copies.

Perception deliberately uses simulation instance labels and virtual multiview depth. Assets are procedural bottles, cups, cartons, glue sticks and toy blocks, with randomized properties and support locations. Verified runs cover seeds 282527379 and 282527381, not an arbitrary-layout success rate. Kitchen, dining table, worktable and tray placement are implemented. Bed/floor placement, tray carrying, hand-to-hand transfers and arbitrary household meshes are not validated in this branch.

Recent failure fixes: preserve grip force/static load compensation; settle actual delivered commands and measured posture; adapt Cartesian descent with torso assistance; measure slip in gripper coordinates through base turns; normalize only sub-microradian joint-limit overshoot; and distinguish bounded placement-search timeout from no feasible support. None removes collision, hold or support checks.

All remaining entries are chronological development history with superseded statuses, paths, process IDs and tuning values.

## Historical checkpoints

Local commits: `b9f2afd7d5`, `8bbe24f35e`, `d83bed46a7`, `43adedb0e4`; not pushed. A fresh fetch confirmed `origin/main` is already an ancestor (135 ahead, 0 behind), so no rebase is needed.

- **Passed native MCP:** `bimanual-v11`, both hand-specific picks followed by independent worktable placements; the other hand retained its object. `delivery-v11` picked and carried the bottle to dining; placement then exposed a planner timeout. `cup-v7` is an earlier successful cup-to-tray run.
- **Current fixes under test:** 25-second precise base settling; short collision-checked body translations/turns bypass room navigation; both arms fold before travel. Rigid collision probes now omit force solving, with matching transforms/contacts in regression and saved apartment checks.
- **Dining transfer:** SDK RRT failures returned an empty list, so the adapter incorrectly reported a trajectory-length error. The adapter now preserves the real failure. Saved-state replay identified a constrained-upright search timeout. A new Cartesian IK transfer preserves cargo attitude and the other hand, checking every joint edge; it finds 34 waypoints in 2.28 seconds in that exact scene. Native execution is queued.
- **Tests:** 47 focused local-positioning/navigation/execution regressions passed, then 14 execution tests passed with the empty-plan guard. Mypy passes all five changed implementation files after adding three existing MuJoCo API signatures to local stubs.
- **Running sequential detached tests:** `local-final-status.json` (carton-v12, cup-v12), then `transfer-final-status.json` (delivery-v12: pick → dining → place → re-pick → kitchen → place). All under `/tmp/r1pro-classical-bootstrap`; only one full stack at a time. Host RAM, not GPU memory, limits parallelism.
- **Still unverified:** full inter-room place/re-pick delivery and language-provider HumanCLI round trip. External agent testing requires the pending user approval; no prompt has been sent. The old desktop ACT process remains untouched. Do not claim arbitrary household assets or all randomized layouts are reliable from these fixed seeds.

Latest native results: `carton-v12` passed automatic hand selection, full kitchen navigation, precise body staging, pickup and kitchen placement. `cup-v12` passed left pickup, a local base turn and tray placement. `bimanual-v13` passed both requested-hand picks followed by independent placements with the new upright transfer planner; the other hold remained intact.

`delivery-v12` picked and carried to dining, then reached the preplace pose with the new transfer planner. The final contact descent hit its 10-second wall-time settling deadline; recovery preserved the bottle and completed. Exact-state replay (`replay-delivery-support.log`) reached force-backed dining support after one fresh 0.5 mm command and four simulation seconds of settling. The timeout is now 25 seconds, with unchanged velocity/position/contact gates and added tracking diagnostics. Fourteen execution regressions pass after that change. Current detached job: `settled-delivery-status.json`, supervisor PID3495692, case `delivery-v14` (same complete six-action delivery/re-pick sequence). All other latest supervisors completed.

Live MCP tools were inspected: independent get_scene/get_surfaces/pick_object/place_object/go_to/stop_action/recover_action/reset_scene/wait_for_action, plus server utilities; no ACT tool is exposed. HumanCLI's external provider round trip remains untested pending approval.

`delivery-v14` again reached dining, but a different redundant joint posture from the Cartesian transfer left the fixed-torso lowering leg just outside IK convergence (1.09 mm error). Current Cartesian segments can use bounded torso assistance with the other hand's pose preserved when the fixed-torso solve fails. Commands retain static preload for all moving joints and preserve gripper targets. The exact saved failing descent now plans in 3.73 seconds with <0.05 rad torso-joint changes. Physics replay and `delivery-v15` are running; status `adaptive-delivery-status.json`, supervisor PID3503769. The previous supervisors completed.

Tracking diagnostics initially included a NumPy boolean that JSON could not serialize; native action execution continued but one navigation evidence file was not written in v14. Diagnostics now cast scalar types, and the execution regression uses NumPy planner endpoints and checks JSON serialization. Fourteen tests and mypy on both changed implementation files pass. No motion/contact success threshold was relaxed.

`delivery-v15` passed pickup, dining navigation, dining placement, re-pick and navigation to kitchen. Its final kitchen placement exhausted the bounded search. Filtering obstructed body poses before applying the IK candidate cap and adding the calibrated neutral posture as an additional endpoint seed found three fully checked kitchen candidates in 20.74 seconds. The best candidate's continuous upright transfer also passes in the saved scene (64 waypoints, 3.22 seconds). `HomeKinematics.solve` accepts an optional explicit seed; default behavior is unchanged, and seed vectors contain only the 18 modeled body/arm joints.

`delivery-v16` exposed a real frame bug in the inherited cargo monitor: it compared world-frame TCP/object offsets across a turn. The bottle remained in the right hand; world offset changed 15.49 mm, but gripper-frame offset changed only 0.64 mm. Inventory now adds `tcp_offset_local`, and slip checks use that field. The existing world-frame field remains available for reachability calculations. Two tests (one per hand) rotate/translate held cargo without reporting slip, then inject actual 20 mm relative motion and require rejection. Twenty-one related state/reachability/execution tests pass. Current native rerun: `grip-frame-delivery-status.json`, supervisor PID3522263, case `delivery-v17`. This is the only full test stack; prior supervisors have completed.

`delivery-v17` completed pick, dining navigation and placement. Its re-pick then stopped because the inactive left elbow was 9.2e-8 rad outside the SDK's conservative limit. Solver outputs and full posture goals now normalize at most 1e-6 rad of numerical overshoot; larger violations and non-finite values are rejected. The exact saved failed goal passes SDK collision-checked planning in 0.024 seconds (`limit-replay.log`). Five boundary regressions and mypy on both changed sources pass. Current detached run: `limit-delivery-status.json`, supervisor PID3536109, case `delivery-v18`, same six-action delivery. No other full validation stack is running.

`delivery-v18` completed the first five actions, including dining release, re-pick from the new position and kitchen navigation. Kitchen placement timed out after 30 seconds. Exact-state replay found its first valid pose at approximately 30 seconds (minimum joint clearance 0.107 rad), then two valid candidates within 60 seconds. The bounded search now permits 60 seconds and reports budget exhaustion explicitly instead of treating it as proof of no feasible support. No geometric or contact threshold changed. The canonical-wrist experiment was diagnostic only and was not adopted. Next native sequence: `budget-delivery-status.json`, case `delivery-v19`.

The entries below are chronological history and include superseded tuning values and completed process IDs.

## Current implementation (under validation; not promoted)

- Separate `r1pro-classical-apartment-sim` / `r1pro-classical-apartment-sim-agent` blueprints. No ACT modules in the graph.
- GraspGenX pinned provider from main, using measured R1Pro 100 mm parallel jaw sweep and correct +Z generator / -Z palm transform. Isolated `.classical-venv` overlays existing runtime packages without changing the active ACT environment.
- New `classical_selection.py` conjunctively matches color, kind and robot-relative side; no silent object/hand substitution.
- `classical_perception.py` samples actual ray intersections from virtual multiview depth with simulator instance IDs. This deliberately uses sim labels and virtual sensor viewpoints, not claimed RGB semantic perception or a real robot sensor arrangement.
- `classical_planning.py` assesses actual generated TCP poses with DimOS Pink IK, whole-body posture, collision sweeps, joint-limit margin and manipulability. Equivalent orientations of symmetric jaws are allowed. Physical object attachments exist only in planning snapshots.
- `classical_skills.py` owns an explicit asynchronous perceive/generate/reachability/position/stage/approach/close/lift/hold state machine, separate placement and navigation skills, cancellation and explicit recovery. DimOS plans posture/local base motion and ControlCoordinator executes Cartesian joint trajectories; KronkNav and holonomic task handle apartment travel.
- Whole-body placement search is implemented but its new path is not yet physically validated in the deployed blueprint.
- Shared provider now exposes `num_samples` and passes `max_candidates` into upstream inference; previously the runtime silently capped at 100 regardless of adapter config. Default remains 200 samples / 100 outputs.
- Shared simulator/coordinator builder accepts `policy_module=None` to create a stack without ACT modules.

## Evidence so far

All artifacts and detached development logs are in `/tmp/r1pro-classical-bootstrap`.

- Real GraspGenX checkpoint inference: 834 current object depth points; 600 real proposals. No fake provider/heuristic substitute used.
- Isolated real-physics test (`execute.py`, `execution-grip-force.log`, `execution-status.json`) **passed**: SDK RRT pregrasp staging, generated grasp, two-finger contact, >10 cm unsupported upright lift, transfer 5 cm to another support position, release and retreat. Final bottle upright, supported, settled, released; placement error 0.00326 m.
- The first test exposed a supported-grasp planning transition: ownership is still null before lifting because the table supports it. Local lift planning now permits the selected object's verified two-pad grasp as an attachment in the snapshot.
- A transfer test exposed a grip-force bug: arm trajectories must preserve commanded gripper closure, not command the measured contact width. The latter unloads gripping force. Fixed; left/right command array order is explicitly left then right, whereas ARMS enumerates right then left. Keep this distinction.
- 36 focused tests passed (combined selection, world-frame segmented depth and existing GraspGenX adapter tests). Tests require local socket access for existing DimOS module lifecycle cases.
- Mypy passed all 10 newly introduced/changed provider files before the latest placement-search changes; rerun after final changes.
- Registry was regenerated by its test. The generator intentionally reports failure for an uncommitted generated diff; verify with CI=1 or after committing.

## Native blueprint tests

- `blueprint-v1`: loaded the real GraspGenX provider and found a feasible requested right-hand grasp. Failed before any motion because local base positioning incorrectly used the apartment travel validation path, which requires a prepared navigation session. Fixed to use the SDK local base task and its physical sweep validator.
- `blueprint-v2`: detached native MCP test currently running (PID 3303295 at launch), isolated MCP port 10026 and Zenoh scout 224.0.0.224:19492. Seed 282527379, actions `pick:right:object_2`, `place:right:worktable`. It includes the new body-aware placement search but predates later cancellation/contact-monitor refinements.
- Source test entry point: `dimos/robot/galaxea/r1pro/demo_classical_apartment.py`. It uses only local MCP, no external language model.
- User desktop ACT instance PID 3255711 was already running and remains untouched. Never stop it as part of test cleanup.

## Remaining before calling this ready

1. Finish and fix native MCP pick/place until both hands pass, including holding another item without unloading its grip.
2. Validate multiple generated object layouts and shape families; isolated bottle success is not generalization evidence.
3. Validate body/torso selection and carrying routes to kitchen/dining, including placement and re-pick.
4. Ensure selected-object contact loss halts transfer and recovery preserves each held object. Validate support before release.
5. Add focused regression coverage for commanded grip preservation, supported-grasp planning and cancellation.
6. Review/format/type-check, update registry, commit without coauthor trailers, record concrete launch instructions. Do not advertise this preview as ready until native evidence supports it.

No new ACT training is running. Long test processes use detached sessions and logs. Public GraspGenX code/model installation was approved by the sandbox reviewer. External language-agent tests have not been performed.

## Follow-up validation and fixes (same continuation, 2026-09-13)

The early `blueprint-v2` run completed right-hand pick + worktable placement. `bimanual-v2` completed left pick, right pick while left held, left placement, right placement through native MCP and the real provider. Those successes predate subsequent fixes; do not claim every current path is validated from them.

Wider tests exposed real remaining problems:

- Seed 282527380: left pick from dining failed the old late furniture-collision validator. Moving apartment contacts into the SDK RRT search fixed that pick (`random-left-v2`); its return/placement still failed docking.
- Seed 282527381: a later run stopped because the measured TCP orientation was outside the Cartesian approach tolerance despite joint errors under 0.02 rad. Torso gravity preload was being lost when stationary joint commands were replaced by measured positions. Local arm segments and gripper-only commands now preserve stationary actuator targets. Staging additionally performs bounded, measured-TCP IK corrections before approaching.
- `PrimitiveSceneState.inventory()` previously reported the right observer's `grasped` flag even for a left-hand hold. It now combines both observers and exposes `grasping_arms`; pick refuses lift without the requested hand's two-pad contact. This is shared, additive state-reporting correction, not ACT retraining.
- Placement now checks force-backed contact with the requested support before release, permitting at most 10 mm additional collision-checked descent. No support means keep the grip and report failure.
- Carrying at an extended pickup posture clipped a wrist camera in the native path corridor. There is now a compact carrying-posture search with fixed grasp orientation, compensated other hand, and actual apartment collision checks. Faster torso motion during one carry caused a dropped bottle (`dining-v4`), which was detected and halted. Loaded torso trajectories have since been slowed to 0.08 rad/s with 0.15 rad/s² acceleration; this latest change still needs native validation.
- Docking now searches several setbacks/lateral positions on a snapshot rather than blocking the physics lock. Carry preparation/recovery protects every held item. No live attachments or object relocation were introduced.
- A geometric jaw-closing check initially rejected an already successful left grasp because the proposal was at the very top of a tilted cylinder. In addition to raw GraspGenX poses, the new search tests body-centered grasp translations for the current symmetric procedural props, retaining the generated orientation and applying full approach/closing/lift checks. This is explicit classical grasp refinement, not a claim of unmodified GraspGen poses or arbitrary asset generalization.

Latest detached batch at this edit: `/tmp/r1pro-classical-bootstrap/verified-status.json`, supervisor PID3359541 at launch. Earlier batch files: `batch-status.json`, `retest-status.json`, `layouts-status.json`, `travel-status.json`. Runs are simulation-only, isolated MCP10026 / Zenoh19492, and do not affect the user's old desktop process. Harness results now record source SHA256 so tests can be tied to loaded code.

34 focused regressions passed after support/contact fusion changes; mypy passed 10 implementation files before the latest centering/loaded-speed edits. Re-run affected checks and native tests before promotion. Navigation and robustness across layouts remain unfinished. No new ACT training is running. No external language-model test has been run.

## Saved checkpoint and latest continuation

- Local preview commit: `b9f2afd7d5` on `feat/r1pro-classical-apartment`; pre-commit hooks passed, no coauthor trailer, not pushed.
- 67 focused tests (including registry and GraspGenX adapter) passed; mypy passed 12 implementation sources at that checkpoint.
- `dining-v5` and `dining-v6` both physically picked and carried a bottle along the native route to the dining table, including the arrival turn, without losing it. This is positive navigation evidence, not yet completed delivery.
- `bimanual-v6` physically placed the left object upright, supported, released and settled while preserving the right hold. Its final `complete` was false because a 10 cm retreat from the new body-centered grasp was 8 mm short of the existing above-object clearance check. The retreat now accounts for object height; the success check was not loosened.
- `dining-v5` placement selected a posture with only 0.2% wrist joint margin. Placement search now rejects less than 4% margin and ranks the remainder by margin as well as base travel.
- Exact `random-right-v5` failure snapshot replay now **passes support, release and retreat** (`support-replay-result.json`). The command-to-measured arm offset was causing a millimetre-level Cartesian descent to undertravel while the geometric planner correctly refused further virtual penetration. Local arm commands now retain measured static load compensation, and motion completion additionally waits for low measured joint velocity.
- Small TCP staging corrections now retain static load compensation and can use torso assistance with the other hand preserved if fixed-torso IK cannot converge. They remain bounded and collision-checked.
- The finer support descent starts from measured TCP, advances 1 mm at a time, and remains limited to 10 mm. Collision diagnostics name the offending geometry and save a replayable state on failure.

Current supervisors: `/tmp/r1pro-classical-bootstrap/arrival-status.json` (v6), then `/tmp/r1pro-classical-bootstrap/integration-status.json` (supervisor PID3391484 at launch, queued v7 bimanual, full delivery/re-pick, carton, cup). Do not mistake an older success for validation of a later source version. Full agent/HumanCLI language-provider testing remains unperformed; local MCP is the verified command path. The new launch instructions are in `dimos/robot/galaxea/r1pro/CLASSICAL_APARTMENT.md`, still marked under validation.


## Contact/route diagnosis and queued validation

Local checkpoint `8bbe24f35e` follows the preview, with preload preservation and height-aware retreat; neither checkpoint is pushed. Further changes below are still under validation.

- `cup-v7` (seed 282527381) completed native MCP cup pick, hold, tray placement, physical support, release and retreat. Both picks passed again in `bimanual-v7` and `bimanual-v8`, but left placement still failed support seeking; do not advertise full bimanual readiness.
- `delivery-v7` completed pick and native dining navigation, then missed the measured preplace tolerance. Its terminal arm velocity was still 0.014 rad/s. The prior endpoint gate allowed 0.03 rad/s, too loose for millimetre contact corrections. It now requires delivered actuator targets and <=0.0025 rad/s for three fresh samples. A focused regression covers both conditions.
- Exact runtime MuJoCo model and qpos/qvel/ctrl snapshots are now saved per action. Support samples and commanded/measured trajectory endpoints are included in action reports. `bimanual-v8` replay showed no initial table contact; two smaller, settled descents established force-backed contact with the correct worktable. The runtime now plans each 0.5 mm descent from fresh measured TCP rather than accumulating unexecuted offsets; at most 20 steps and 10 mm measured travel, with contact/collision checks retained.
- Candidate search now removes near-duplicate generated grasps, prioritizes body poses that center the target in the selected arm's workspace, and considers upright placement yaw changes. A recorded dining search that previously found no placement now found three feasible poses in 15 seconds. Native execution remains the acceptance gate.
- Empty arms had stayed extended during kitchen travel. The saved native route clipped furniture at those arms. A collision-checked compact empty-arm posture makes the same route pass the full-body sweep in replay (`fold-diagnosis.log`); loaded-hand behavior is unchanged.
- 11 classical skill regressions passed after the endpoint/contact changes; mypy passed the seven classical sources before the last endpoint/empty-arm edits. Re-run final checks after native verification.

Current detached supervisor is `/tmp/r1pro-classical-bootstrap/settling.py` (PID3427032 at launch), status `settling-status.json`. It waits for `tracking-status.json` (delivery-v8) then runs bimanual-v9, carton-v9, delivery-v9 on private MCP10026/Zenoh19492. No ACT training, external LLM test, or user desktop-process changes. Logs and results live under `/tmp/r1pro-classical-bootstrap`; each result records loaded source hashes.


The saved `delivery-v8` scene exposed an over-conservative joint filter: requiring 4% of every joint range rejected a whole-body solution whose smallest absolute clearance was 0.1086 rad (>6 degrees). Placement now requires at least 0.06 rad absolute clearance (twice the maximum accepted compensation offset), while still ranking normalized margin. The old 0.2% wrist-stop pose remains rejected. The same saved scene now yields three complete placement/retreat plans in 5.25 seconds (`place-margin-result.json`).

`bimanual-v9` has completed both hand-specific picks and the left placement, preserving the right hold. The left item reached force-backed worktable support after 0.70 mm measured extra descent, then released and retreated successfully. The right placement is still running at this edit. No readiness claim until remaining native tests finish.

The workstation `.env` is an ignored symlink to the user's existing `/home/mustafa/dimos/.env`, so the separate demo CLI reuses configured provider credentials/settings without requiring shell exports. No credential values were printed or copied into version control. No external model request was made.


`bimanual-v9` right placement then exposed a separate adapter edge case: SDK RRT returned a single waypoint for an already-reached posture, while the trajectory generator requires two. The classical executor now explicitly represents that single-pose hold with two identical points; the measured TCP staging check still runs. Added regression coverage. The whole bimanual sequence must be repeated with this fix.


73 focused tests (classical skills/planning/perception/selection, shared physical state, navigation, provider, blueprint registry) passed. Mypy passed all seven classical implementation files. A parallel full-stack test was attempted after checking GPU headroom, but host RAM/swap became the actual bottleneck: 31 GiB RAM and all 15 GiB swap used; Zenoh watchdog overruns and a 66-second response to a 20-second wait caused the kitchen harness to time out. That run is not evidence of a completed kitchen trip. Its native full-body route did pass and navigation began. The extra stack exited; subsequent native tests are strictly sequential.

`bimanual-v10` stopped during gripper closure and explicit recovery completed successfully. Closure contact oscillations were about 0.003 rad/s, so precise 0.0025 rad/s settling now applies to staging/approach/support descent; grasp closure and other moves use 0.01 rad/s plus physical contact/hold verification. All phases still require delivered actuator targets, fresh simulator state, and completed trajectories.

Queued sequential checks: `final-status.json`, supervisor PID3444321 (`final_checks.py`), waits for delivery-v9 to finish and then runs bimanual-v11, carton-v11, layout-v11 with explicit recovery-on-failure. Private MCP10026/Zenoh19492. The prior second-port bimanual-v10 process has exited. Do not run another full model stack concurrently on this 31 GiB host.


`delivery-v9` exposed that folding only the occupied hand is insufficient: the empty hand can still clip furniture. Carry preparation now folds both hands, preserving each measured grasp transform. The exact rejected dining route passes with that posture in replay (`fold-loaded.log`, three SDK waypoints, native route full-body sweep passed). This is planning replay evidence; physical loaded transport with the new both-arm posture is queued for validation after the current sequence.
