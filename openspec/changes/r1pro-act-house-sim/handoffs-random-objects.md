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
