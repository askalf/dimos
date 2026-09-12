# Reusable ACT manipulation primitives

The requested behavior is `pick(object, arm)` → held object → independent
`place(region, arm)`. A language agent composes these actions only when the user
asks for them. "Pick" ends with a grasped, lifted object, without release or an
inferred destination. A destination-bearing instruction can compose pick and place.

## Current implementation

`r1pro-objects-sim-agent` exposes `pick_object`, `place_object`, measured inventory,
stop, recovery and reset. The first command stops after current two-pad contact,
a 10 cm lift and a verified hold after cancelling ACT. Placement is a new rollout
from measured held state, recomputes free tray space, and verifies supported release.
A full tray does not prevent picking; it can refuse placement while preserving the hold.

These commands reuse the existing full-sequence checkpoint. Pick still supplies a
legacy tray-goal context; this is a transitional execution adapter, not proof that
the model learned an independent or destination-invariant pick primitive. Current
capabilities remain right-hand table picks and tray placement. Tray unloading,
left-hand picking, arbitrary supports and carrying a held random object through
the house remain unimplemented here. The original bottle/navigation blueprint is
separate and does not automatically confer those capabilities on this blueprint.

## Target architecture

1. Resolve an object and an arm from measured scene state. An explicitly requested
   arm must be preserved. Automatic arm selection scores feasible candidates.
2. Select a stable, collision-free object pose within the requested placement
   region. Account for the object footprint, obstacles, support and available space.
   The agent names a region; it does not invent low-level joint targets.
3. Ask the locomomanipulation layer for a base/torso pose that puts the source or
   destination within that arm's validated policy workspace. Check physical IK
   reachability, collision clearance and demonstrated coverage separately.
4. Execute the selected ACT pick or place policy through ControlCoordinator.
   Policy output must own only the intended arm and agreed torso resources;
   a policy must not open or reset another hand already holding an object.
5. Check current contacts, lift, support, release and non-target motion. Maintain
   held-object state per arm. Failures stop or recover explicitly and never become
   success simply because a trajectory finished.

Use a small family of policies conditioned on object/goal geometry and the chosen
arm, rather than a policy for every object, order or named room. Whether arms share
one canonical policy or use two adapted checkpoints depends on measured transfer;
the earlier naive mirroring test failed and is not a deployable left-arm skill.
Changing the room should be handled by navigation and local-frame goals, provided
the resulting grasp/place is within the policy's trained geometry and workspace.
This is not a guarantee of generalization to new shapes or support heights.

## Reusing the existing demonstrations

`demo_segment_objects` creates pick/place manifest views of the original RGB files:

```bash
python -m dimos.robot.galaxea.r1pro.demo_segment_objects \
  --source recordings/r1pro-act-task/jobs/random-objects-act-v1/collection \
  --output recordings/r1pro-act-task/jobs/my-primitive-data
```

Pick includes above/approach/grasp/lift and the recorded lift hold. Place starts at
clear_sources from the already-held state and includes supported release/retreat.
The converter uses the same frame slice for images, actions, state and statistics.
Episode boundaries prevent action chunks from crossing from pick into place.
Original images, demonstrations and weights are not rewritten. Keep paired views
from a layout on the same train/validation side; they are not independent evidence.

The prepared views are `jobs/random-objects-primitive-data-v1`: 115 pick segments
(17,480 frames) and 115 place segments (53,490 frames). They preserve the old
right-arm observation/action contract as preparation inputs. Before training a
new arm/goal contract, explicitly migrate the feature mapping and normalization,
then warm-start compatible weights. ACT task text alone is not a substitute for
an implemented conditioning input.

Add targeted demonstrations for held-state variation, tray-source grasps, varied
supported destinations and left-arm use. Retain old data as rehearsal. Fine-tune
new artifacts and evaluate independent pick-hold, place-from-hold, pause/resume,
wrong-arm rejection, full tray, and mixed requested sequences on fresh seeds.
Do not schedule another long run on unchanged data just to repeat the same loss.

## Navigation tuning

The user permits higher yaw speed for future navigation tests. The current planar
servo and holonomic profile cap yaw at 0.12 rad/s. Tune both the task profile and
servo/slew limits together, verify loaded-object stability and swept collisions,
and keep the final approach precise. No navigation speed changed in this primitive
interface update; locomomanipulation integration remains a separate workstream.

## Independent-policy implementation (September 12)

The next generation has four explicit contracts: pick/right, place/right,
pick/left, and place/left. Each commands seven arm joints and that hand's gripper.
The torso and the other hand remain observation context, with no action ownership.
Pick has no destination features. Both wrist cameras are real rendered views;
left-arm weights receive an explicit signed initialization, followed by training.

`object_primitive_task.py` supplies an offline SDK teacher. It prepositions the
physical base before each primitive, then keeps base and torso commands fixed.
`object_primitive_state.py` measures ownership separately for each gripper and
protects the other hand's cargo relative to its TCP during base motion.
`placement_regions.py` selects empty supported regions with footprint and open
finger clearance, returning no candidate when full. This currently has physical
bench coverage for table/tray regions; arbitrary house surfaces are not validated.

Evidence before learned-policy evaluation:

- `jobs/primitive-teacher-pilot-10`: both arms × table/tray source = 4/4 physical
  pick/hold/place checks, with placement back onto the original support.
- `jobs/independent-primitives-pilot-v1/collection`: 32/32 separately accepted RGB
  primitives across eight fresh layouts, including tray-to-table transfers.
- `jobs/primitive-bimanual-pilot-01`: both left-first and right-first sequences
  picked two objects, held both, and placed them separately with the SDK teacher.
- Right-pick and left-place checkpoint migration loaded successfully and produced
  finite eight-action chunks. A 20-update optimizer smoke test passed.

These teacher and migration results are not claims of ACT success. The existing
`r1pro-objects-sim-agent` still uses the previously validated transitional policy.
Independent checkpoint promotion and native interactive validation remain pending.

The detached job is `recordings/r1pro-act-task/jobs/independent-primitives-act-v1`.
It adds 24 layouts to the eight-layout pilot, trains each arm/primitive for 2,000
updates from the existing trained checkpoint, then evaluates chained ACT on fresh
seeds starting at 330000. Original weights and demonstration files stay intact.
Both pick and place must pass physical checks; an ACT pick failure skips placement.
Artifacts are saved separately and never automatically replace the working demo.

Monitor from the worktree:

```bash
cat recordings/r1pro-act-task/jobs/independent-primitives-act-v1/status.json
tail -n 5 recordings/r1pro-act-task/jobs/independent-primitives-act-v1/run.log
```

The runner survives terminal disconnects. Its status identifies the current stage
and stage log; heartbeat updates are every five minutes. `command.json` records
the invocation, and completed stages/checkpointed training can be resumed.

Locomanipulation review used main `9314c76543` and PR #4096 head `849b284fc5`.
The PR adds a feedback base trajectory task and splits whole-body plans between
base and joint tasks, with coupled cancellation. The bench teacher currently uses
its own physical prepositioning for collection. Native integration must use the
SDK execution boundary, retain the plan ID, and verify completion before ACT.
The PR is still open and its API may change; it has not been merged into this branch.

## Native integration under validation

Development is currently in `/tmp/dimos-r1pro-primitives` on
`feat/r1pro-primitives-integration`. It includes main `9314c76543` and the five
commits from PR #4096. The original training checkout is unchanged while its
job runs. The existing `r1pro-objects-sim-agent` remains the earlier working demo.

The new `r1pro-primitives-sim` and `r1pro-primitives-sim-agent` compose four
independent ACT runtimes. Each arm's trajectory task claims only its seven joints
and gripper. Source selection includes tray contents. `pick_object` stops after
current grasp/lift verification and a five-second hold; `place_object` requires an
explicit region. The SDK plans base prepositioning and executes the retained plan
ID through PR #4096's feedback base task. The simulator measures the final pose
before allowing ACT to start. No teacher grasp is called by these commands.

`define_placement_region` registers a rectangle contained within the measured
physical worktable. Placement checks space for the object and open fingers and
chooses the nearest available point to that region's center. This provides new
placement regions without new policy IDs. Other support heights and arbitrary
house surfaces remain outside this first primitive bench validation.

`recover_action` preserves a confirmed hold or opens only supported contacts and
uses the SDK to restore the empty failed arm. It keeps a failed recovery latched;
only a verified recovery or explicit scene reset clears that requirement. An ACT
failure remains a failure even when a later recovery succeeds.

Evidence as of this integration checkpoint:

- Main merge: 121 focused tests passed. PR #4096: 53 tests passed.
- Primitive selection, ownership, runtime discovery and recovery: 30 tests passed.
- Seven native implementation files passed mypy.
- `/tmp/r1pro-primitive-native-base-04`: SDK base execution completed within 3 mm.
- `/tmp/r1pro-primitive-native-act-02`: MCP right-hand pick of object_2 passed,
  including the five-second hold with no placement.
- `jobs/primitive-native-interactive-01`: pick passed; 20-step ACT placement
  reached the tray upright but kept its gripper closed and timed out.
- `jobs/primitive-native-interactive-steps30`: using separate checkpoint copies,
  pick, tray placement and ACT tray unloading passed. Subsequent table placement
  still timed out with the hand closed near the support. This is not an accepted
  end-to-end result; placement tuning and mixed-hand ACT evaluation remain open.

All checkpoint originals are preserved. These results do not promote the new
blueprint over the previously validated object demo. The new native test runner is
`demo_primitive_interactive`; it accepts explicit `pick:arm:object_id` and
`place:arm:region` actions and records each real MCP outcome. Long evaluations are
launched as detached jobs with a `pid`, `command.json`, `run.log` and `result.json`.

## Current training and verification

The first four independent policies have completed 2,000 updates each. On the
same eight held-out arm/layout cases, executing 20 actions from each chunk passed
1/8 complete sequences. Executing all 30 actions passed 8/8 picks and 7/8 complete
pick/place sequences. This changed inference configuration only. One right-arm
placement still failed; these eight cases are a small validation set, not a
reliability estimate for arbitrary objects or two-hand operation.

`jobs/primitive-place-refine-v1` now fine-tunes the existing placement policies
with original demonstration rehearsal and new placements starting from actual
ACT-held poses. Pick weights are preserved. The same-profile initializer was
checked against the saved checkpoint: every model tensor and normalization
mean/std tensor was unchanged before optimizer updates. New exports use 30-action
chunks and are evaluated on fresh seeds beginning at 352000.

The first correction batch accepted nine right-arm and seven left-arm examples;
rejected grasps and disturbed-object episodes are excluded. A separate left-arm
supplement adds verified examples before left placement conversion. Its original
manifest and all source arrays are retained, with supplement provenance recorded.

Monitor the active work from any terminal:

```bash
watch -n 10 cat /home/mustafa/dimos-wt/r1pro-act-sim/recordings/r1pro-act-task/jobs/primitive-place-refine-v1/status.json
```

The independent native verification job waits for refinement, then tests a
right-arm tray cycle, a left-arm tray cycle, and two objects held simultaneously:

```bash
watch -n 10 cat /home/mustafa/dimos-wt/r1pro-act-sim/recordings/r1pro-act-task/jobs/primitive-place-refine-native-v1/validation/status.json
```

Both jobs run detached; closing the terminal or stopping `watch` leaves them
running. They save failures and never automatically replace the working demo.
The native verification uses local MCP without an external LLM request.

Recovery test `jobs/primitive-native-recovery-02` passed: a deliberately interrupted
left pick returned only that arm home, preserved all object placements and the
other hand, and cleared the recovery requirement. The preceding failed test is
retained. The primitive blueprint explicitly selects the SDK's shared RRT-Connect
planner over the same RoboPlan collision world: the native RoboPlan RRT rejected
the recorded, collision-free, in-range arm state while the shared planner found
and executed a valid path. Normal pick and place commands still use ACT.
