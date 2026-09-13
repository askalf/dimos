# Random-object ACT execution handoff

## Current state — 2026-09-12

The user requires ACT grasps for four or five randomly generated objects, and sparse checks while long jobs survive terminal disconnects. Work is in `/home/mustafa/dimos-wt/r1pro-act-sim`. The shared root checkout and its environment are untouched.

**Latest steering:** the agent must expose primitives: a bare pick means grasp/lift/hold, and placing requires an explicit destination instruction. The desired future interface supports choosing either arm, picking from surfaces or trays, placing into feasible regions, and locomomanipulation positioning the robot inside the learned skill's workspace. Do not concatenate an unsolicited full sequence. The user permits increasing yaw speed during future navigation tests. Preserve the existing data and weights; no new training has started.

**Primitive interface implemented:** `pick_object` now stops at current two-pad contact plus 10 cm measured lift and verifies the object remains held after cancellation. `place_object(destination="tray", arm="right")` is a separate rollout from held state and recomputes free placement space. A second pick cannot overwrite a held object. The agent composes both only when the request explicitly specifies placement. Right-arm table picks / tray placement remain the current scope. This is reuse of the old checkpoint with separate termination conditions, **not independently trained, location-general primitives**. The old model still consumes tray-goal context during picking; retain its usual free goal when available and use non-reserved context if full. The planned transition is in `dimos/robot/galaxea/r1pro/OBJECT_PRIMITIVES.md`.

Primitive validation and data preparation:

- `jobs/random-objects-primitives-01`: first pick → five-second hold → explicitly requested place passed, and the second pick completed. The test then hit an HTTP keepalive-close race at a five-second idle boundary. The manual validator now uses fresh HTTP connections and does not retry motion POSTs. This was a client failure, not a simulation grasp crash.
- `jobs/random-objects-primitives-02`: three separate pick/hold/place cycles passed. The fourth pick missed/tipped its target and reported failure; no placement followed. Explicit reset passed. This run used synthetic empty-tray context even when a real free goal existed, which unnecessarily changed the learned grasp input.
- `jobs/random-objects-primitives-03`: **all four picks, five-second holds, separately commanded placements and explicit reset passed** for seed 210000, object order 3 → 1 → 4 → 2, with normal free-goal context preserved. This is one development layout, not evidence of arbitrary source/destination or left-arm generalization. All primitive validation jobs have now finished; no training or demo process is intentionally left running.
- 55 focused geometry, physics, recovery, command and phase-view tests passed. Current two-pad holding is checked separately from historical grasp success, including a physical release regression.
- `jobs/random-objects-primitive-data-v1`: 115 pick views (17,480 frames) and 115 place views (53,490 frames), referencing the original compressed RGB recordings without modifying/copying them. The split is immediately after the recorded lift/hold, before clear_sources. These are reused segments, not 230 new demonstrations. Source manifest SHA256: `3dd4736961b5069ec04e453397b345cd11d27feb7761bbfc43c2c6c3a2602f08`.
- One segment of each kind converted through real LeRobot (152 pick frames / 459 place frames); frame counts and action mean/count were verified against exactly the selected source slice. Image/state/action reads and normalization all use the same bounds. A new arm/goal schema and its weight/normalization mapping still need explicit design before training. Do not mistake new task text for a new ACT conditioning input.

The user also wants right/left hand choice. Left-hand ACT remains outstanding, not completed by rejecting it. The failed mirrored transfer experiment is preserved below. Household mesh expansion remains later.

**Interactive implementation:** `dimos run r1pro-objects-sim-agent` plus `dimos humancli`, or `r1pro-objects-sim` plus direct MCP commands without an LLM. See `dimos/robot/galaxea/r1pro/OBJECT_INTERACTIVE.md`. Native scene generation happens at build time, with a unique output directory and shared scene/home/limits for control. The tools are `get_scene`, `pick_object(object, arm)`, `place_object(destination, arm)`, `wait_for_action`, `stop_action`, `recover_action` and `reset_scene`. No automatic pick sequence runs. IDs/shape/spatial selectors operate on eligible source objects; unsupported left requests cause no movement.

**External-language validation is awaiting explicit approval.** Automatic approval review rejected an attempted live-agent test because it would send a natural-language command and simulator/tool context to the configured external model using existing API credentials. An async question asks to approve one command: “Pick the nearest object with your right hand and put it in the tray.” Do not bypass that rejection or infer approval from elapsed time. Local MCP/ACT tests use no external model. The validator now uses the actual `/human_input` and `/agent` Zenoh streams, including final replies and tool results. Its recorded-response mode passed; the real external-model mode has not run.

Native interactive evidence:

- `jobs/random-objects-interactive-01`: initial startup race; commands arrived before .6 simulated seconds. Fixed by waiting for settled home before selecting.
- `jobs/random-objects-interactive-02`: requested indices 2,0,3 physically completed; index 1 timed out, then classical supported release/retreat returned home. The failed pick remained reported as failed. Explicit reset passed.
- `jobs/random-objects-interactive-recovery-02`: deliberately limited picks to three seconds. Nearest (object_2) timed out and recovered; rightmost (object_1) then timed out and recovered without an intervening reset. Explicit reset passed. Earlier recovery-01 exposed nanoradian overshoot of the gripper stop; trajectory starts/goals now account for numerical noise using actual joint limits.
- `jobs/random-objects-interactive-cancel-01`: stop during an ACT attempt reported cancelled, held without automatic release, and explicit reset succeeded.
- `jobs/random-objects-desktop-01`: native GLFW viewer and object_3 ACT pick passed.
- `jobs/random-objects-agent-playback-01`: local recorded-response agent test passed through HumanCLI's actual input/response streams, McpClient, MCP skills, and physical ACT execution. It called get_scene → pick_object(object_3, right) → wait twice → get_scene, then published a final reply; object_3 was physically inside the tray and the action completed. This validates wiring, not language understanding. RECORD was unset; no external model was used.
- `jobs/random-objects-interactive-h30-01`: unchanged weights, 30 actions per inference, seed 210000, requested object_3 → object_1 → object_4 → object_2: **4/4 physical picks and explicit reset passed**.
- `jobs/random-objects-interactive-h30-02`: same 30-action deployment, seed 210006, requested object_3 → object_4 → object_1 → object_2: **4/4 physical picks and explicit reset passed**. These are development layouts previously evaluated offline, not fresh acceptance seeds.
- New interactive default: `recordings/r1pro-act-task/policy-objects-interactive`, exported from the refined checkpoint with `prepare_r1pro_deployment --action-steps 30 --objects`. Weights unchanged; the original 20-action artifact is preserved. Broader 30-action development results remain 11/12 single picks and 2/8 complete scenes; do not imply arbitrary-layout reliability from the two native passes.

Recovery planning uses a copied MuJoCo state and SDK IK. It refreshes FK after mj_step and applies follower-finger equality in kinematic collision checks. It never moves live qpos, grasps as a fallback, drops an unsupported object, retries ACT or resets silently. Runtime checks preserve all objects and stop on obstacle contact. Eighteen focused command/recovery tests pass, including real supported-contact recovery and airborne refusal (15 command tests plus three physical recovery tests).

Left-arm reuse investigation is preliminary. `jobs/arm-symmetry-check/result.json` finds approximate arm symmetry with joint signs [1,-1,-1,1,-1,1,-1], with about 2.6 mm maximum TCP reflection error in 20 sampled poses. A separate `jobs/left-coordinate-transfer-01` experiment mirrored coordinates/cameras/actions using the unchanged right-hand ACT weights in a mirrored tabletop scene. It did not establish a bilateral grasp (0/1). It is not a deployed capability and does not prove transfer is impossible. New arm-conditioned or canonical-arm demonstrations/validation remain necessary. The prototype is `/tmp/r1pro_left_transfer.py`; its data are separate from the working policy.

The original 10,000-update pilot scored **6/12 single picks and 0/8 complete scenes**. Refinement completed another **30,000 updates on those same 115 demonstrations**, taking 97m35s and scoring **9/12 singles and 1/8 complete scenes** (14/21 attempted sequence picks). There were no classical grasp interventions. The current checkpoint is `recordings/r1pro-act-task/jobs/random-objects-act-refine-v1/policy`; its deployment default remains 20 actions per inference. Do not promote it as reliable full-scene or agentic random-object support.

A completed execution-horizon comparison reused the same saved model and development seeds, without training:

| Actions executed per inference | Single picks | Complete scenes | Successful sequence picks |
|---|---:|---:|---:|
| 20 (checkpoint default) | 9/12 | 1/8 | 14/21 |
| 30 | 11/12 | 2/8 | 14/20 |
| 10 | 7/12 | 0/8 | 13/21 |

Evidence: `jobs/random-objects-act-horizons-v1/comparison.json`. These small development sets do not establish acceptance or justify silently changing the saved artifact. Refinement and horizon supervisors have finished; read each job's `status.json` and exit marker for status instead of relying on an old PID.

Native Zenoh + ControlCoordinator integration now has a standalone validation runner, `dimos.robot.galaxea.r1pro.demo_object_packing_stack`. The selected-object run in `jobs/random-objects-native-01` physically passed for seed 210000, index 2. The complete native sequence in `jobs/random-objects-native-02` placed indices 2, 0 and 3; index 1 timed out without lifting. ACT stopped cleanly. Existing SHM resource-tracker teardown warnings were also observed. The newer registered interactive blueprint and recovery evidence are described above; full learned sequence reliability and left-hand ACT remain pending.

The start/occupancy variation physics pilot completed **23 accepted / 24 attempted** demonstrations across 12 layouts. It perturbs the initial right-arm posture by up to .02 rad and preloaded tray occupants by .012 m, preserving geometric clearance. One approach failed at seed 120007, target 1; all failures are retained. This was a no-image demonstrator check, **not 23 additional trainable RGB episodes**. Results: `jobs/random-objects-variation-smoke/manifest.json`.

The demonstration generator uses the SDK's Pink IK with a weak current-posture task (`posture_cost=1e-5`) and an inward joint-limit margin. It favors continuity near the preceding pose, not a fixed neutral elbow/torso posture. ACT rollout itself commands joint targets; it does not run this teacher IK. Do not add a competing posture controller during ACT execution without validating the resulting behavior. Stronger demonstration posture preferences should be tested separately before changing the training distribution.

## Implementation and actual evidence

- `object_packing_scene.py` generates independently positioned, sized, coloured and weighted boxes, cylinders and compound bottles. There are four or five objects, with randomized partial occupancy. Bounded whole-layout restarts preserve source finger clearance. A local overlay gives the new task a 29 cm internal tray and enough release clearance for short objects; the existing bottle scenes are unchanged.
- `object_packing.py` defines 52 simulator-ground-truth features: selected object relative to TCP, destination relative to TCP, object orientation/dimensions/shape, home relative to TCP, and sorted neighboring poses/extents with padding masks. There are also two RGB cameras and 20 measured joints. ACT outputs all 20 joint targets. This is a bounded primitive-shape distribution, not arbitrary unseen household categories.
- `object_packing_task.py` uses the new manipulator SDK for offline demonstrations only. Learned evaluation never calls teacher actions or IK. Success requires physical bilateral grasp/lift, upright supported release, containment, settled motion and return home; non-target motion/collisions are checked. Geometry and contact support are separate predicates so tiny momentary support-force changes do not falsely report spills.
- Teacher pilots 01–06 exposed arm/furniture contacts, infeasible torso routes and insufficient short-object release clearance. Pilot 07 passed **15/15** physical picks across eight layouts, all three families and 0–3 initial occupants. Image smoke passed **4/4**, followed by successful conversion, initialization, five-update fitting, export and a deliberately short learned execution. That smoke did not prove learned success.
- Full collection in `jobs/random-objects-act-v1`: **115 accepted / 118 attempted**, 64 layouts, up to two selected targets per layout, with 43 cylinders, 39 boxes and 33 bottles. Each successful pick is an independent episode; this is not an enumeration of full-scene permutations.
- Pilot training completed **10,000 updates**, batch 32, about five dataset epochs, in **32m14s**. Diagnostic loss fell from .0379 at 2500 updates to .0239 at 10000; that loss is not the acceptance metric. Paired choices from the same layout stay on the same side of the diagnostic split.
- Initial evaluation attempts crashed when a successful pick returned `numpy.bool_` to JSON. `pick_complete()` now returns a Python bool, with a real physical grasp + JSON regression. Incomplete reports are archived under `eval-*-incomplete*`; use only the current complete `eval-single/result.json` and `eval-sequences/result.json`. The corrected evaluation finished at **6/12 and 0/8**. The pipeline now rejects incomplete evaluation reports instead of treating partial totals as a completed evaluation.
- Recorded-state prediction diagnostics are saved in `fit-diagnostics.json`: typical active-joint MAE .003–.016 rad, with some held-out lift errors up to .027 rad. Additional fitting is a hypothesis being tested, not an established fix. Current physical seeds (200000+ and 210000+) are development evaluations; freeze a new final test set before acceptance.

## Incremental training and verification

- `prepare_object_act.py` can initialize from an already-trained, matching object profile without reinitializing its environment projection. Same-width but incompatible profiles are rejected.
- Warm starting also preserves the original observation/action normalization. LeRobot otherwise replaces saved processor statistics with the new dataset statistics when fine-tuning, which can change physical predictions before the first update. The initializer saves the new dataset's original statistics in `normalization-before-warm-start.json`, then uses the checkpoint's mean/std. Pass a **newly converted dataset**, never the archived baseline dataset. The source checkpoint is read-only.
- A small local ACT regression proves bit-exact physical action predictions before/after warm-start preparation despite deliberately different new-data statistics. Both initializer tests pass. Mixing additional image demonstrations with old episodes and choosing the next training budget still require a deliberate follow-up; no expanded-data run has been launched.
- Shared `ObjectPackingState` provides the same read-only geometry, goal vector and physical evidence to native/offline execution. The physical regression compares both monitors without allowing metadata queries to change qpos/qvel/ctrl. Both nominal and perturbed initial postures pass.
- The geometry/physical tests passed (30 tests), and the generated blueprint registry check passed (6 tests) after its expected regeneration. Strict typing is checked separately in native and isolated LeRobot environments.

Previous commits `14f90296c6`, `cf2346856c`, `a1137b2c81`, and `775f92712b` were pushed to `origin/feat/r1pro-act-sim` without coauthor trailers. This follow-up adds the registered interactive blueprint, exact object/hand intent, supported-contact recovery, the validated 30-action deployment default and native agent-stream playback validation. The older full-sequence validation jobs have finished. See the primitive validation entries above for current work; no training has been launched. Recordings, weights, downloaded assets, logs, `.venv` and local data links are excluded from commits.

Main integration remains pending: a rebase tried to replay 92 dependency commits and conflicted with main's relocation of the isolated Python runtime. It was aborted cleanly; `backup/r1pro-before-random-objects-20260911` preserves the previous branch. Do not claim this branch is rebased or force-push it casually.

## Later household-asset work

The existing HSSD package includes a real `flower_mug` mesh and 12 convex collision pieces. Its local mesh is about 7.4 × 10.5 × 8.2 cm; world AABB dimensions must not be mistaken for local geometry. Other local assets include a whiskey bottle, bowl, journal and camera. Scans of a small wind-up dog, shark toy, tape roll, scissors and coffee mug were downloaded to `recordings/r1pro-act-task/household-assets`, with their Google Scanned Objects source metadata/license files retained. These are not integrated, grasp-qualified or used in training. Do not replace their geometry with disguised bottle colliders or shrink them silently to fit the hand. Preserve handles/cavities through appropriate collision decomposition, and establish grasp/support frames before changing the observation contract.

## Next work

1. Preserve the current image dataset and refined checkpoint. Continue improving repeated selected-object execution and explicit failure recovery; avoid another long fit on unchanged data without evidence.
2. Validate native requested-target selection, stop-when-full and repeated picks. Two native four-object sequences now pass at 30 actions per inference, but broader sequence reliability remains limited; do not call arbitrary layouts accepted. Keep ACT as the grasp executor.
3. If collecting more training data, add targeted start-state/approach/occupancy corrections with images, retain the original demonstrations, and fine-tune a new artifact from the saved weights. Keep source artifacts and normalization intact; freeze a fresh final test set before acceptance.
4. Revisit actual household geometry after the current version is stable, per the user's latest steering. Current primitive data remain useful rehearsal/pretraining but are not demonstrations of handles, thin stationery or irregular toys.
5. Integrate current main carefully. The earlier multi-stop unloading/regrasp SDK QP failure remains separate; existing tray carrying remains classical.


## 2026-09-12 — independent primitive policies underway

User authorized the next phase: independent pick/place, either arm, tray unloading,
general placement regions, and asked to inspect main plus locomomanipulation PR
#4096 (subject to change). Main fetched at 9314c76543; PR head 849b284fc5. No rebase
or PR merge has happened yet. Root checkout /home/mustafa/dimos was not edited.

Added arm-only profiles (8 commands; measured torso/other arm as context), removed
pick destination inputs, explicit ACT weight/normalization migration, separate
RGB pick/place collection, physical region selection, fresh chained ACT evaluation,
and a persistent training runner. New per-arm measured ownership protects another
hand's object through prepositioning. Native policy declarations exist; their
interactive blueprint integration remains to be completed and validated.

Physical SDK teacher evidence: primitive-teacher-pilot-10 passed 4/4 table/tray
source checks across both arms. independent-primitives-pilot-v1 accepted 32/32 RGB
primitives on eight layouts, with separate episode boundaries and initial states.
primitive-bimanual-pilot-01 passed both arm orders: pick first, pick second with
first still held, place second, place first. These are demonstration results, not
learned ACT validation. Earlier pilot failures are retained as diagnostics.

Actual checkpoint migration succeeded for right pick and left place; both loaded
with eight-action outputs. Six isolated migration tests passed. Fourteen contract/
placement tests and the 28 existing layout tests passed (a missing action-key
construction issue was fixed). A 20-update ACT optimizer smoke passed. Native
six-file mypy and the three additional state/binding modules passed. The combined
primitive/layout/existing interactive regression run passed 60 tests, and all six
blueprint-registry checks passed. Isolated learner final typing also passed for all three modules.

Detached process 1436126 runs jobs/independent-primitives-act-v1. command.json and
pid/status.json/run.log are in that directory. It collects 24 additional layouts,
reuses the successful eight-layout pilot without copying source arrays, converts
and warm-starts four policies, trains 2,000 updates each, exports separately, and
runs eight fresh chained ACT cases (four seeds × two arms) starting at 330000.
The source is policy-objects-interactive (the existing trained weights); no old
weights/data were rewritten. It has stage heartbeats every five minutes and does
not auto-promote artifacts. Do not run another uv learner concurrently with the
runner's learner stages: the isolated project environment is shared.

Remaining: inspect learned evaluation, collect other-hand-held context if needed,
wire native primitive skill/state/prepositioning, integrate the SDK base execution
boundary, verify house regions and recovery, and test mixed interactive commands.
The existing object blueprint remains the previous working version. Prior pending
permission for an automated external LLM/API test is still unanswered; local ACT
and recorded-response MCP checks do not require that API call.

### September 12: main/PR integration and first native independent ACT checks

Continue in `/tmp/dimos-r1pro-primitives`, branch
`feat/r1pro-primitives-integration`. Main merge commit is `5c83ed2b02`;
PR #4096 integration ends at `cb5c3d4fc5`. No Codex co-author trailers. These are
local commits; original `feat/r1pro-act-sim` is still at `607ce29498` while the
four-policy training job runs from that original checkout. Root Alfred work and
its environment were not changed. Do not switch the training checkout before
its detached pipeline completes.

New native implementation, tests and `OBJECT_PRIMITIVES.md` describe the exact
state. Native SDK base motion passed. Independent right ACT pick/hold passed via
MCP. At 30 executed actions per chunk, ACT tray placement and tray-source pick
also passed. Table placement still stalls with the gripper closed: do not claim
full success or replace the old demo. The 20-step and 30-step comparison jobs are
`jobs/primitive-native-interactive-01` and
`jobs/primitive-native-interactive-steps30` in the original recordings directory.
Next: finish both left policies' detached training, evaluate fresh layouts,
resolve release stalls, validate mixed-hand holds and recovery physically, then
promote only checkpoints passing those checks. The long pipeline remains
`jobs/independent-primitives-act-v1` (parent PID 1436126), with `status.json` and
per-stage logs. Do not start another learner against its isolated project while
it is training. The integration checkout has its own isolated learner project.

### September 12: chunk comparison, placement refinement, native recovery

Continue in `/tmp/dimos-r1pro-primitives`. The first native integration was
committed as `dfde4e567`. Original four-policy training finished. Its automatic
evaluation failed on a missing optional `roboplan` dependency, not on training.
The original failure status and partial output were archived; resuming from the
integration runner skipped completed stages and evaluated the intact exports.
Baseline 20-action chunks: 1/8 full sequences, right pick 4/4. Comparison in
`jobs/primitive-baseline-steps30`: 8/8 picks, 7/8 complete sequences at 30 actions,
with unchanged model weights. One right placement still failed. Existing native
30-action testing also has a table-release stall. No new checkpoint is promoted.

Active detached refinement: `jobs/primitive-place-refine-v1`, parent PID 1551849.
It preserves the four original exports, rehearses original place data, initializes
from each existing eight-action place checkpoint without changing model tensors or
normalization coordinates, and fine-tunes place only for another 2,000 updates.
The tensor/normalization preservation smoke passed. Correction collection starts
from an actual ACT grasp, then records SDK placement; no failed grasp is replaced
by a teacher grasp. First batch: right 9 accepted/3 rejected; left 7/5.
`jobs/primitive-place-left-supplement-v1` (PID 1576929) collects eight more left
attempts, preserves the initial manifest, and supplements it before left
conversion, with an explicit stage guard against changing consumed training data.
Check its status and `collection/supplements.json` before reporting final counts.

`jobs/primitive-place-refine-native-v1` (PID 1577650) waits for refinement and then
runs local MCP right, left, and simultaneous-held-object sequences. Its status is
`validation/status.json`. Results stay separate and no external LLM is called.
Refinement evaluates unseen seeds 352000..352003; native checks use seed 340000.
Do not equate successful training, teacher examples, or an accepted tool call
with physical ACT success. Other-hand-held RGB training remains absent; the queued
mixed-hand check determines whether additional data is needed.

Recovery now passed physically in `jobs/primitive-native-recovery-02`: forced
one-second left-pick timeout, explicit recover, only the failed arm returned home,
objects/other hand preserved, recovery latch cleared. The first test's SDK native
RRT threw "Invalid start configuration". Offline diagnostics verified no model
collision or joint limit violation; the SDK shared RRT planned the same state.
The primitive blueprint now explicitly selects RRTConnectPlannerConfig with the
same RoboPlan world and SDK execution. No global planner behavior was changed.

Focused refinement/skill tests passed 21 cases, and model/normalizer warm-start
smoke passed. Native/refinement typing checks passed; subsequent commit hooks
provide the final source checks. Run status commands are in OBJECT_PRIMITIVES.md.
Remaining: inspect refinement and queued native outcomes; improve the remaining
failures, validate arbitrary supported regions and both-hand contexts, then
promote explicitly. House delivery/general support heights remain outside the
new primitive bench's validated scope. The original interactive demo is retained.

Supplement finished successfully: seven additional left-arm examples were
accepted, bringing correction counts to right 9 / left 14, each alongside the
30 original placement demonstrations. The initial seven-example left manifest is
retained as `manifest-before-supplement.json`; supplemental arrays are referenced
by absolute path and not copied or rewritten. Right fine-tuning is running;
left fine-tuning and automatic DimOS/MCP validation are queued.

The user asked what "SDK" / "native" meant. Clarified that all commands use DimOS:
ManipulationModule, ACT runtimes, ControlCoordinator, and the blueprint/MCP
interface. RoboPlan is a dependency inside DimOS; only the demo planner selection
changed to DimOS RRTConnectPlanner with the existing RoboPlanWorld. Use specific
DimOS module names in further explanations to avoid implying another robot stack.

Original `/home/mustafa/dimos-wt/r1pro-act-sim` was fast-forwarded from 607ce29498
to integration through 4d4a08e18b after the original four-policy pipeline finished.
Its source and documentation now include these changes; active jobs continue from
`/tmp/dimos-r1pro-primitives`. Root `/home/mustafa/dimos` Alfred work is untouched.
Recent source commit: 58fdec4749; documentation clarification: 4d4a08e18b. Both
passed commit hooks without co-author trailers. These changes are still local.

### Current-state preview requested

Prepared `recordings/r1pro-act-task/policy-primitives-preview` as a complete,
separate copy of all four first-round independent policies. Verified every weight
checksum unchanged; only n_action_steps is changed to 30, with provenance in
preview.json and each deployment.json. The two new primitive blueprints now use
this preview path, so `r1pro-primitives-sim` opens an idle full MuJoCo view with
direct MCP commands. Original checkpoints and the old object demo are untouched.
Known baseline result remains 7/8 offline sequences, with placement failures;
this does not promote the unfinished refinement. See OBJECT_PRIMITIVES.md for
minimal desktop launch and separate pick/hold/place commands.
At this check right placement refinement/export finished and left placement is
training. The local DimOS/MCP validation queue still waits for all exports and
held-out evaluation. Preview sessions use default ports, separate from the
queued verification session (MCP 10016, discovery 19479).

The exact CLI check caught a registry omission: top-level calls to the custom
blueprint factory were invisible to the AST scanner. Added terminal
.global_config() calls, regenerated all_blueprints.py with its pytest generator,
and added regression tests resolving both names through get_blueprint_by_name.
This fixes dimos run discovery; factory-level/MCP tests alone had missed it.

### September 12: refinement results and carried-object collision fix

Both placement refinements and their evaluations finished. Refined ACT on fresh
seeds 352000..352003: 8/8 picks, 4/8 complete sequences. Do not compare this directly
with the baseline's 7/8 on different seeds 330000..330003. No promotion occurred.
DimOS/MCP evaluation `primitive-place-refine-native-v1/validation` finished:
right pick/place-to-tray/unload all passed, then table placement timed out with
closed jaws; left pick passed but tray release timed out; both-hand sequence
passed right pick but left pick missed while the right hand held its object.
Thus independent pick/hold, tray unloading and the interface exist, but full
placement reliability and other-hand-held training remain unfinished.

The user wanted HumanCLI, not only direct MCP. Added its exact command beside
`r1pro-primitives-sim-agent` in OBJECT_PRIMITIVES.md. Both CLI names already
resolve in the original worktree through 3d0a07a7af. Agent credentials must be
configured by the user; external LLM testing remains unapproved. Local MCP and
physics tests require no external model. Estimated roughly one hour for the next
fix/test pass, explicitly not a promise of full reliability within that hour.

Diagnosed seed 352002: base prepositioning, before ACT place, carried object_3
through object_5. The old check moved only the robot and ignored cargo. Added a
held-object-only mode to the existing PlanarTransport: both measured held objects
move in the planning copy; the tray and unheld objects remain obstacles. The
primitive scene plans a detour with 5 mm sweep sampling. Offline teacher and
DimOS execution share this route. DimOS still plans/executes every waypoint via
ManipulationModule and the coordinator, and its actual returned base trajectory
is checked again with current cargo before execution. Existing carried-tray mode
retains its behavior. No live object attachments or teleports were introduced.

Saved failing physics state: `jobs/primitive-transfer-probe-01/before.npz`.
Replaying it with the new route retained the grasp; unrequested-object movement
was zero except 0.00000078 m numerical settling on object_1. The selected object
traveled 0.388 m. Focused transport/primitive tests: 29 passed; mypy passed all
six changed production files. This alone is not an ACT placement success claim.

Detached follow-up `jobs/primitive-cargo-validation-v1` (parent PID 1676858) runs
the full DimOS pick/place on seed 352002, then baseline and refined checkpoints
on that same seed and both arms. Its `status.json`, `dimos/result.json`, and
baseline/refined reports preserve separate outcomes; no checkpoint promotion.
Source work remains in `/tmp/dimos-r1pro-primitives`; the user's preview is not
changed by background model evaluation. Remaining priorities: inspect this run,
address ACT release fixed points, add demonstrated other-hand-held contexts,
validate arbitrary supported regions, then promote only passing policies.

The cargo validation's full DimOS case has now PASSED: right ACT pick object_3,
measured base routing, then ACT placement into the tray on seed 352002. Both
actions report success without recovery. Matched-seed offline comparison is
still running. This verifies the integrated route and actual DimOS trajectory
validation, not just an offline geometric plan.

Saved the fix as eb2246f66f and fast-forwarded the original R1Pro worktree to it;
commit hooks passed and no co-author trailer was added. The two matched checks
starting at seed 352002 each passed 2/2, but that evaluator chooses table/tray
tasks by the layout's index in the run, so this shortened run tested table
placement, not the original index-2 tray task. A full matched comparison is now
detached as `jobs/primitive-matched-eval-v1`, parent PID 1689153: baseline and
refined policies, both arms, seeds 352000..352003 in their original task order,
30 actions, updated collision routing. Check its status.json and each result.json.
No new training is running and no model was promoted.

### September 12: actual interactive failures and next context refinement

Remote branch checked directly with `git ls-remote`: last push remains
389632263b8be793b2de8ed7f9cfb2cb6d855761 at 12:34:56 Pacific. That checkpoint
validated four right-hand table pick/hold/explicit-tray-place cycles on seed
210000, order 3→1→4→2. It was a limited usable demo, not general independent
policies. No rollback was requested; keep this remote checkpoint intact.

The user tested the primitive preview. Session
`recordings/r1pro-primitives/24701af314824a5a8de2287affc5f51e` has failed right
picks of left-side object_1, a successful right pick of object_2 followed by
failed placement at (0.599, 0.489), and a failed left pick of right-side object_2.
Recovery succeeded for supported failed picks but correctly refused to release
or reset when the selected object lost stable support. Do not execute the
earlier interrupted "pick the light blue bottle" request in the live session.

Full matched evaluation finished: original 5/8, refined 7/8 on identical seeds
352000..352003 at 30 actions with cargo routing. The remaining refined failure
was an obstructed prepositioning goal. This does not establish the preview's
cross-table, left/right interchangeability, or two-hand reliability.

Training had a real context gap: single-sided scenes and the other arm parked.
The preserved normalizer gives each inactive-joint context dimension std=0.01;
holding postures therefore lie far outside the original examples. New optional
`--interactive-context` collection shares the exact bilateral layout function
with the simulator. It collects both arms, sources on either side and in the
tray, half with another object held, and placements across the worktable.
Teacher setup grasps are explicitly teacher actions, not reported as ACT.
Separate pick/place boundaries and other-hand ownership checks remain intact.
Refinement now optionally trains all four existing profiles, preserves weights
and normalization, rehearses original and previous correction data, and requires
at least four successful other-hand-held examples plus eight total new examples
per profile before training. Existing placement-only refinement mode remains.

Base positioning no longer clamps forward translation to zero: a far-table
goal now stays in the learned local arm workspace. The physical planner checks
the nominal pose and nearby ±4 cm alternatives, with a shared ten-second route
budget. Placement selection tries other empty points when their carrying pose
is obstructed. All executed base trajectories still pass the MuJoCo cargo check
and run through DimOS ManipulationModule/ControlCoordinator. No arm grasp/place
fallback has been substituted for ACT.

Demonstration probes are in `/tmp/primitive-interactive-teacher-smoke` (8 picks,
7 placements before wider targets), `/tmp/primitive-interactive-wide-smoke`,
`/tmp/primitive-interactive-placement-smoke`, and the latest
`/tmp/primitive-interactive-routing-smoke`. Intermediate versions exposed
blocked carrying poses; retain their failed reports. The final probe is still
running at this note. Far-table DimOS ACT test is `/tmp/primitive-far-table-dimos`.
Regression tests passed 37 cases and changed-source typing checks passed.
Extended validation now covers both arm orders, cross-table requested-arm picks,
tray unloading and a custom far-table region. Model promotion remains pending.

Committed this work locally as edd1b099d5 with all hooks passing and no co-author
trailers. The full DimOS far-table test PASSED both ACT actions: right pick of
object_2 then placement in the explicit region centered at (0.60, 0.48), width
and depth 0.12 m. It used the existing refined policies, with no new training or
classical arm fallback. This verifies the positioning fix for that actual user
case. Final wider teacher smoke: right 4/4 picks and places, left 2/4 picks and
places; two left cases could not establish the other hand's setup hold because
its base workspace was obstructed. Those failures remain rejected, not training
examples. Broader collection must satisfy its minimum other-hand-held count.

Detached job is now `jobs/primitive-interactive-refine-v1`, parent PID 1799748.
It collects RGB on 24 new layouts 360100..360123, then fine-tunes all four
existing policies for 2,000 updates each, using original pick demonstrations
and the previous refined place demonstrations as rehearsal. Rehearsal manifests
only reference original arrays; nothing was erased or restarted from scratch.
Initial source model is `primitive-place-refine-v1/policies`; fresh offline
evaluation uses 361000..361003. Keep the source/model files stable while it runs.
Its `command.json`, `source_commit`, `rehearsal_sources.json` and `status.json`
record exact provenance. Normalization coordinates are preserved on warm-start.

Detached extended DimOS validation is `jobs/primitive-interactive-refine-native-v1`,
parent PID 1799749, waiting on that job. Its status is `validation/status.json`;
it uses MCP 10016 and Zenoh discovery 19479, seven cases including both arm
orders, requested cross-table hands, unloading, and the far-table region.
No external LLM is called and no checkpoint is automatically promoted.
The source integration branch has advanced locally; origin/feat/r1pro-act-sim
remains at 389632263b. Do not claim the new policies are ready until evaluated.

### September 12: interactive refinement completed; positioning regression fixed

`primitive-interactive-refine-v1` finished all four 2,000-update refinements.
Collection accepted 35 pick/place pairs: 17 right, 18 left; 14 pairs included
an object already held by the other hand. Previous datasets and initial models
remain intact. No new policy was promoted to the preview.

Its fresh offline evaluation completed 3/8 sequences on 361000..361003.
Four cases (361000 and 361002, both arms) failed during base positioning before
ACT started. All four picks that actually ran passed; one right-hand table
placement stalled with the jaws closed. These seeds differ from the earlier
7/8 result, so the raw totals do not establish a model regression.

The extended local HTTP MCP/DimOS evaluation completed 3/7 full scenarios.
Requested right-hand and left-hand cross-table pick/place cases both passed,
as did right pick followed by custom far-table placement. In the longer right
sequence, loading the tray and picking the object back out passed, but the
final table placement stalled closed. The ordinary left pick and both two-hand
orders failed: missed/unstable grasps or insufficient lift. One left grasp
briefly met the lift condition then slipped immediately after stopping ACT;
the completion guard correctly rejected it. The evidence does not yet show
whether stopping contributed to the unstable grasp. Logs and action histories
are in `primitive-interactive-refine-native-v1/validation`.

Reproduced the four base failures using the original saved scenes. Every
nominal/nearby pose collided with the table/tray, while standing farther back
was clear. Expanded candidate search to include 48--52 cm forward target reach,
retaining the existing nominal and nearby choices, full collision checks, and
ten-second planning limit. This keeps forward positioning for far-table goals.
All four previously blocked cases now physically preposition successfully;
reports are `/tmp/primitive-route-diagnosis/results.json` (before) and
`fixed.json` (after). Fourteen transport regression tests passed, including
both-arm coverage for this case; changed-source typing and lint checks passed.

Next verification is a detached matched evaluation in
`jobs/primitive-positioning-matched-v1`: previous place-refinement policies
versus the new interactive-refinement policies, same 361000..361003 layouts,
same task ordering, 30 action steps, and the corrected positioning search.
This is evaluation only. Check its status.json and each result.json; do not
interpret a completed training job as an approved interactive release.
Left grasp stability, two-hand sequences and reliable release remain open.

### September 12, 18:52 Pacific: matched results and failure traces

The matched positioning evaluation completed: previous policies 3/8 full
sequences, interactive-refinement policies 4/8. Neither set had any initial
base-positioning failures. The new policies passed 5/8 picks and 4/5 attempted
placements. Remaining new-policy failures: right pick on 361000, both picks on
361002, and left table placement on 361003. That last placement released the
object but did not satisfy all physical placement criteria; it is not the
closed-jaw failure seen in the earlier integrated test. Do not conflate them.

No training is running. Started detached diagnostic
`jobs/primitive-failure-traces-v1`, which compares ACT and the demonstration
controller on those exact four failing tasks, with per-step actions, measured
joints, TCP/object geometry and initial/final MuJoCo snapshots. For the placement
case, both controllers start after a fresh ACT pick and identical placement
selection. This comparison is diagnostic only, never an ACT fallback in the
interactive skills. Its status.json, results.json and individual trace.json
files retain the evidence. Preview artifacts and the remote checkpoint remain
unchanged. Next: inspect grasp-centering/lift behavior and the exact placement
criterion that fails before choosing further correction data or tuning.

### September 12, 19:02 Pacific: reproduced ACT centering and placement errors

Failure traces completed. All four demonstration-controller cases passed;
all four corresponding ACT cases reproduced their failures. Their measured
initial states match, including the ACT-established hold before placement.
The three failed grasps approach with roughly 1--2 cm forward TCP error, then
close on the object edge and lose contact or tip the object. The teacher stays
centered and retains a stable lift. The left placement does release and retreat,
but the object ends near (0.500, 0.581) instead of (0.530, 0.540), about 5 cm
away and outside the requested region once its footprint is considered. The
teacher places near (0.528, 0.542) and passes. Do not weaken the success criteria
or describe that particular failure as a gripper-release stall.

Started detached `jobs/primitive-feedback-eval-v1`, parent PID 2197824.
Before another training round, it evaluates the same eight cases at 10 and
20 executed actions per observation against the established 30-action 4/8
baseline, with unchanged weights and physical checks. If either reaches at
least 7/8, it copies that candidate into the job directory, changes only its
execution-window setting, verifies the remaining artifact files are identical,
and runs the seven extended local DimOS/MCP cases on port 10017 / Zenoh 19480.
No external LLM is called, no preview is overwritten and nothing is promoted
automatically. If neither qualifies, the evidence will guide corrective data
collection. No training is currently running. Overall demonstrated reliability
remains 4/8 offline and 3/7 integrated until new results establish otherwise.

### September 12: feedback rejected; corrective approach demonstrations

Feedback comparison completed: 10-action execution passed 1/8 sequences and
2/8 picks; 20 actions passed 2/8 sequences and 3/8 picks. Neither qualified
for the integrated-test gate. Retain the 30-action setting (4/8 sequences).
More frequent observation updates did not solve these policies' grasp errors.

Added an offline corrective collector, `demo_collect_primitive_approaches`.
ACT performs the approach until it is about to close an open hand near an
upright, settled, supported object. The DimOS teacher then centers, closes,
lifts and holds; only teacher-generated frames become labels. Unsafe approach
states are rejected. Corrected picks must retain a five-second physical hold.
Place demonstrations remain separately labeled. The shared bilateral scene
retains an object in the other hand for half the layouts, with ownership and
disturbance checks throughout. No teacher fallback was added to runtime skills.

Table-source collection can prefer 48 cm reach to cover the failing posture;
the existing collision-checked candidate set and runtime default order remain
unchanged. The preference only reorders demonstrated, bounded reach candidates.
New `--approach-corrections` refinement mode trains all four existing profiles
with original-data rehearsal and unchanged normalization, after checking the
minimum accepted data and other-hand-held counts. It is mutually exclusive with
the previous interactive-context collection mode; both older modes remain.

Initial physical smoke `jobs/primitive-approach-smoke-v1` passed 8/8 corrective
pick/place pairs (four per arm), including four pairs with another object held.
Its table positioning used the original preference. The longer-reach smoke is
`jobs/primitive-approach-smoke-v2` on 370200..370203 and is running separately.
Saved arrays contain only approach/grasp/lift/hold in pick labels, and separate
place/release/retreat phases; ACT approach prefixes are metadata, not labels.
Twenty data/profile/refinement tests and fifteen transport tests passed; all
four changed source files passed typing, and lint/format checks passed.

Next detached pipeline is `jobs/primitive-approach-refine-v1`: gate on the
longer-reach smoke, collect 32 fresh layouts 370000..370031, preserve all data
through rehearsal from `primitive-interactive-refine-v1/merged`, and fine-tune
the four current interactive policies for 4,000 updates each. Compare old/new
on identical fresh seeds 371000..371003, then run the seven integrated cases.
Nothing is automatically promoted or pushed. Check the pipeline's status.json
for its actual running stage; collection/verification is not completed training.

### September 12, 22:05 Pacific: three profiles trained; disk interruption resumed

The longer-reach smoke passed its acceptance gate. Full correction collection
retained 41 physically verified pick/place pairs: 18 right-arm and 23 left-arm,
including 19 pairs with an object held in the other hand. Failed demonstrations
were excluded. Existing rehearsal data and all source policies remain intact.

Right pick, right place and left pick each completed 4,000 refinement updates.
Their final checkpoints and completed-stage markers are saved. The pipeline
then stopped before exporting left pick because free disk space fell below
its 12 GiB reserve; the status retained the previous stage name, so this was
not a failed or lost left-pick training run. Left place had not started.

Space subsequently recovered without deleting this project's datasets or
checkpoints. A waiting uv cache prune was canceled because another live demo
was using the cache. With 31.27 GiB free, resumed the same detached run.py as
PID 2461022. Completed stages are skipped; export-pick-left is now running.
The interrupted status and resume record are preserved in the job directory.

Remaining stages are left-place conversion/training/export, the matched
eight-case held-out evaluation, and seven integrated DimOS/MCP cases. The old
policies scored 3/8 on this round's fresh 371000..371003 baseline; the new
policies have not yet been evaluated. Earlier 4/8 results used different seeds.
Preview artifacts and the remote branch have not been promoted or replaced.

### September 13: apartment reachability integration, not yet a working ACT demo

User requirement remains an interactive randomized apartment with both grasping
and placement on ACT, explicit or automatically selected hands, independent
pick/hold and place/release, and DimOS locomomanipulation/navigation between
supports. The latest direction adds physical reachability plus policy coverage
to hand/stance selection, and intersects supported empty placement regions with
that reach. Do not substitute classical runtime grasps or placements.

Previous `primitive-approach-refine-v1` finished all four 4,000-update profiles.
On matched seeds 371000..371003 it passed 8/8 picks but only 3/8 complete
pick/place sequences (baseline also 3/8). Integrated cases passed left,
right_cross_table and left_cross_table; right, both hand orders and custom_region
failed. No promotion. Remote last verified 389632263b remains the older limited
right-hand checkpoint. User worktree is still at 0187ef1208; new apartment work
is in `/tmp/dimos-r1pro-primitives` and is not yet the user-facing preview.

New apartment stack uses the actual HSSD package, Zenoh, full static pointcloud
into KronkNav, and a ControlCoordinator holonomic task. Local HTTP MCP navigation
trial `jobs/apartment-navigation-v8` passed the complete route to dining_table
and preserved all unrequested objects. Keep heading through the narrow passage
and turn at arrival. Native routes get at most 10 cm interior offsets, exact
endpoints retained, and full swept robot/cargo collision checks before execution.
Circular clearance above .35 m disconnected this route; do not increase it
blindly. Current test speed is .12 m/s, yaw .12 rad/s; speed tuning remains.

New prop families: bottle, hollow cup, drink carton, glue stick and toy block.
These are small procedural household variants within the existing size envelope,
not arbitrary mesh assets or proven general-purpose everyday-object grasping.
Every launch samples a seed. Five items are placed on measured physical patches:
two on worktable, one on dining table, two on kitchen counter. Objects/tray remain
physical, unheld objects do not follow the base. Cup support sums contact force
per support/pad before thresholding, so many small contacts count correctly.

Reachability searches copied MuJoCo state using DimOS model/IK. It checks stance,
arm approach/lift/retreat and held cargo, ranks free hands while preserving an
explicit hand, and reports measured training coverage separately. ACT receives
only the selected hand; torso/other-hand positioning uses the DimOS whole-body
planner. Preparation uses a physics snapshot and commits selection only after
route validation and an unchanged-state check. Recovery can preserve a stopped
navigation pose when all cargo/unrequested objects remain intact. No implicit
reset or release. Full runtime grasp success is still checked physically.

`primitive_workspace.py` audits first-frame target/torso bounds from verified
manifests. Latest policies cover roughly x=.40--.52 m, y=+/- .32 m for picks,
z=.752--.790 m and a nearly fixed torso. Standing farther back allows counter
reach that standing closer does not. Both hands have feasible counter corridors
with torso assistance, but these are OUTSIDE the demonstrated workspace. Do not
confuse IK feasibility with policy competence. Floor/bed/general meshes remain
unfinished. Current SDK planar-base execution interfaces also match open PR4096;
main was fetched (f01a8eed65) and is nine commits newer than this worktree's base.
Rebase is still pending; do not modify the user's Alfred/Go2 worktree.

Three native apartment ACT attempts with existing weights failed on the first
pick: `apartment-pick-baseline-v1`, `apartment-pick-local-context-v1`, and
`apartment-reach-pick-v1`. The last correctly chose the right hand for auto but
timed out with the gripper open. Masking neighbors farther than .8 m did not
solve it. This local observation contract is now used in apartment collection.
The default `policy-apartment-preview` does not exist and must not be presented
as a working launch command yet. No external LLM calls were made in these tests.

Detached collection `jobs/apartment-appearance-v2`, parent PID 2706379, is running
32 layouts, seeds 380132..380163, two source choices, both hands, with other-hand
holds on half the layouts. It collects the source worktable poses in apartment
appearance, not yet the raised dining/counter postures. At seed 380136 it had
3 right pairs and 3 left pairs plus one extra left pick; successful other-hand
held examples exist on both sides. Rejected physical demos stay excluded.
`apartment-appearance-v1` was explicitly paused after poor source sampling;
retain its data and paused.json. Do not resume it with the changed scene contract.

`demo_refine_primitives --collection` can reuse this completed collection with
original-data rehearsal, preserving every old dataset/checkpoint. It requires
at least eight new accepted episodes and four other-hand-held examples per
profile. No fine-tuning watcher has been launched yet at this entry. Need train,
held-out apartment rollouts and integrated local MCP validation before promotion.
Training learner uv mutations and native policy launches must not overlap.
Collection in .home-venv is independent and may overlap training.

Space cleanup: removed only inactive generated dimos-manip-docs/.direnv and ran
Nix GC older than seven days; freed 5,419 paths / 3,149 MiB, preserving active
profiles, datasets and checkpoints. Did not change root .venv. A native build
changed Cargo.lock's lcm-msgs qualification; this is a build artifact, not an
intended source change. Do not include it in commits.

### September 13, 02:30 Pacific: current main, first apartment training batch

Rebased the isolated integration branch with merge topology retained onto
origin/main f01a8eed65. The old merged-main resolution was reused, then only the
nine newer main commits were applied; two adapter docstring conflicts retained
the existing explanation of active transport selection. No user Alfred/Go2
files changed. Apartment checkpoint is now 810b1e0912. Backup ref:
`chore/r1pro-before-apartment-main-rebase-20260913` (145ffe5147).
After rebase, 63 blueprint, Zenoh, holonomic and apartment skill/navigation tests
passed. Focused physical/placement/primitive tests also passed (30 tests before
the latest release-probe regression). User worktree and remote remain unchanged.

Detached pipeline `jobs/apartment-appearance-refine-v1`, PID 2750703, now takes
an immutable first-batch manifest snapshot as soon as the existing eight-new /
four-other-hand-held gates pass. Full source collection PID 2706379 continues
independently through all 32 layouts. No source arrays are copied or removed;
the snapshot references accepted files by absolute path, retains their hashes,
and checks numeric finiteness, frame counts and separate pick/place phase sets.
`data_audit.json` passed: right pick/place 9/9 episodes, left pick/place 11/10;
each profile includes four examples with the other hand occupied. This is not
39 complete sequences: picks and placements are separately counted.

The matched old-policy BENCH baseline on seeds 380200..380203 passed 8/8 full
sequences. This does not establish apartment success. Right-pick fine-tuning is
now running, 4,000 updates with prior-data rehearsal and unchanged low learning
rates, then right place, left pick, left place, bench comparison and four local
apartment MCP cases. Pipeline uses only local MCP, no external LLM calls, and
never promotes automatically. Read status.json and data_audit.json for actual
state. Do not launch native policy stacks while its learner environments mutate.

Reach checking now proves a continuous fixed-torso Cartesian descent/retreat,
checks SDK self-collision as well as copied MuJoCo scene/cargo contacts, and
seeds torso positioning from nearby demonstrated starts when available. Explicit
hands remain fixed. Raised supports can require standing farther back and a
different torso. Physical reach is still distinct from learned competence.

Found and fixed a placement-probe bug: mj_forward does not enforce coupled
finger equalities. Opening only the driver left the follower finger closed in
the planning copy, falsely blocking every retreat. The probe now moves the
follower with its driver while preserving the measured compliance offset.
A regression test performs a real simulated grasp, checks place feasibility and
verifies that live joint/object state never changed; it passed. No contacts or
success requirements were disabled. Gripper yaw is now included in placement
clearance; apartment search first checks object fit then actual hand sweeps.

New offline `--apartment --apartment-reach --interactive-context` collection
samples initial robot stances at the worktable, dining table and counter. Initial
robot pose assignment is training episode setup only, before any grasp or labels;
props remain supported and unheld. Pick/release labels use actual arm controls.
Local placement prepositioning physically moves the base with the grasp held,
then records the independent place skill. This is not runtime teleportation.
Higher-support demonstrations are still being debugged; do not start a large
collection or deploy their policies without a passing smoke.

Manual counter diagnostic: both hands established physical grasps; right also
placed 2.5 cm away, left's shifted destination failed IK. Smoke v1 finished with
six picks and zero placements; v2 was paused and preserved. Besides the coupled
finger bug, a loose reach IK orientation tolerance admitted poses the stricter
teacher could not track. Reach tolerance is now tighter. MuJoCo transforms also
need refreshing after a physical base move before the strict SDK FK check;
BimanualPrimitiveTask.switch_arm now does this. Current detached smoke is
`jobs/apartment-reach-smoke-v3`, PID 2764323, seeds 381000..381001, three choices.
Its running process predates the final switch_arm FK refresh, so inspect results
and rerun into a new output if that old mismatch appears. Failed states now get
small diagnostic snapshots and remain excluded from accepted data.

Apartment transport monitoring now follows the same 20 Hz cadence as primitive
monitoring instead of doing expensive full inventory checks every 500 Hz physics
step. The full swept route check remains. Agent prompt asks for one recovery
inspection after a failed action requiring recovery, without retrying the grasp,
changing the hand/target, releasing an unsupported object or resetting progress.


### September 13: physical reach smoke and serialized incremental training

Updated 2026-09-13T02:58-07:00.

Reachability/collection fixes are saved as `7957c165b7` on the rebased development branch. The 27 focused reachability, placement, workspace, apartment-skill and refinement tests passed; changed production files passed mypy. The subsequent rehearsal-context gate tests passed 9/9. User checkout and preview weights have not been replaced.

Physical smoke `apartment-reach-smoke-v4` (PID 2779835) passed right-hand pick and placement at worktable, kitchen and dining_table on seed 381000. Left pick passed all three; left worktable/kitchen placement passed, while left dining placement rejected its fixed-torso corridor and preserved the hold. Seed 381001 is still running. These are physical demonstrations, not ACT evaluation results. Failed states and rejected data remain saved separately.

`apartment-appearance-refine-v1` (PID 2750703) finished right-pick fine-tuning and moved to right-place. It will train/export all four profiles, run bench comparisons, then four local MCP apartment checks. No external LLM is called.

Two detached continuations are queued under the shared recordings jobs directory:

- `apartment-reach-collection-v1` (PID 2788456): waits for completed v4 smoke with at least one accepted pick/place for each hand at all three supports, then collects seeds 381100 through 381107, three choices per layout and both hands.
- `apartment-reach-refine-v1` (PID 2790331): waits for the previous training AND its native checks to finish, and for appearance/reach collection completion. Freezes unique new accepted examples with source hashes and finite-frame/primitive-boundary checks. Requires at least eight new episodes and two per support per profile; retains prior occupied-hand rehearsal. Fine-tunes existing weights for 4000 updates per profile, then runs six hand/support MCP cases plus a carry/re-pick sequence across dining and kitchen. It records results and does not promote a preview automatically.

Incremental refinement now checks occupied-hand coverage in combined verified rehearsal and corrections; it no longer requires every new-height batch to reproduce four fresh occupied-hand examples. This reuses previous data, but does not claim two occupied hands generalize to new heights: that remains a required runtime check. Duplicate or unsuccessful examples and batches with fewer than eight new examples are still rejected.

Default scene package resolution is valid in the user worktree via its existing `dimos/data/scene_packages` symlink. The temporary development worktree now has the same untracked data link; no user path was hardcoded into production configuration. Training and long collection remain detached; do not start native policy stacks while learner environments are being updated.


### September 13: release clearance and measured carry state

Updated 2026-09-13T03:10-07:00.

The saved left dining rejection had two distinct causes. First, requiring a second 12 cm lift before placement exceeded the wrist workspace by about 2.6 mm. Placement now uses an 8 cm transfer/retreat baseline with full neighbor clearance retained; pick still requires its 10 cm physical lift plus command margin. Second, local transport reused the pick-completion height predicate. The held bottle reached 99.9 mm lift, so it was reported lost despite upright bilateral pad contact. `ObjectPackingState.carrying()` now checks the actual unsupported grasp independently; offline placement/transport use it, while pick completion remains strict.

The saved left dining case now physically passes base adjustment, placement, supported release and retreat (`/tmp/r1pro-dining-place-final.log`). A regression physically lowers a grasp below the pick milestone, then successfully places it; it passed. Changed production files pass mypy and ruff.

Independent DimOS RRT-Connect posture planning, full MuJoCo sweep checks and physical execution also passed at both kitchen approach poses. Max measured joint errors were 0.00151 rad right and 0.00227 rad left (`/tmp/r1pro-apartment-posture-results.json`). This is offline posture validation, not a native ACT end-to-end result.

Smoke v4 completed with picks 12/12, placements 9/12. Its left dining coverage gate correctly prevented bulk collection. V4 data remains untouched. Corrected `apartment-reach-smoke-v5` (PID 2802943, corridor contract 5) repeats seed 381000 across all three supports and both hands. Bulk collection was restarted waiting on v5 (PID 2802944). The later refiner was restarted while waiting (PID 2806466); previous scripts/status are retained as `run-before-v5.py`, `status-before-v5.json` and the corresponding dual-held revision files.

The final queued native suite now also includes `two-held-two-stops` on held-out seed 381204: pick separate worktable objects with right and left, navigate to dining, place right, navigate to kitchen, place left. All eight native cases and bench results must be reviewed before preview promotion. Current source remains in `/tmp/dimos-r1pro-primitives`; no user checkout/preview replacement or push has occurred.
