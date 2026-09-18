# OpenYAM and R1Pro demo inventory

Audit date: September 17, 2026, after the user requested a pause for a PC reboot.

This is the consolidated handoff for a new conversation. It separates demonstrated behavior, experimental work, source branches, and local model artifacts. It supersedes old “currently running,” “current main,” or “ready” statements where those conflict with the evidence below. This inventory was made by reading Git, source, deployment manifests, saved results, and job status files. No simulation or training was restarted for this audit.

## Human-readable overview

**We built a working, constrained ACT bottle-and-tray demo and several successful classical mobile-manipulation scenarios. We did not finish reliable ACT pick/place for arbitrary objects and locations.** The latest open-space performance changes are still being validated. Moving to a newer demo did not mean that it inherited the earlier demo’s reliability or all of its features.

A branch is a version of the source. A worktree is the directory holding that version. A blueprint selects the modules to run. A policy artifact is a separate set of local model files. Several blueprints live on the same branch, and changing a branch does not automatically select or update its policy.

### What happened, why we moved on, and what we have

| Stage / main branch | What it does and how | Why we moved to the next stage | Status and evidence |
| --- | --- | --- | --- |
| **Dual OpenYAM** — hackmit/t7-resume | Two stationary arms use classical planning to put bottles in a bin. A separate ACT bonus learns a right-arm task. HumanCLI selects classical skills. | Wanted a mobile robot in the apartment, not stationary arms. | Classical bimanual local MCP: **5/5**. A user-driven live agent run also completed both bottles and returned both arms home. ACT: **0/10 stable deposits at each of 10k, 20k and 30k checkpoints**. Training completed; learned success did not. |
| **R1Pro single bottle + tray** — feat/r1pro-act-sim | ACT picks one bottle and deposits it. Early mobile prototype moved a fixed/floating tray; the replacement used a free tray, a support surface, and physical two-handed classical carrying. | Wanted five bottles, neat packing and a complete delivery. | Bottle ACT: **20/20 tabletop**, **10/10 house**, and **10/10 free-tray adaptation** in their respective tests. Two complete physical tray deliveries recorded. These are different scene/checkpoint evaluations. |
| **Five-bottle home demo** — feat/r1pro-act-sim | Geometry selects an accessible bottle and empty tray slot; ACT executes bottle-to-tray. Classical two-hand tray handling, KronkNav and a holonomic controller deliver it to the laptop table. Automatic and HumanCLI variants exist. | Wanted arbitrary selection order, mixed objects and independent pick/place requests. | Strongest constrained ACT demo: **18/18 offline scenes / 90 bottles**, **6/6 native packing runs**, two earlier complete deliveries, plus a full-window Zenoh delivery at **0.6 m/s**. Arbitrary bottle order remained unreliable. |
| **Random-object ACT, right arm** — feat/r1pro-act-sim | Selected-object/goal geometry plus images condition ACT on boxes, cylinders and bottles. Initially it always picked and placed. Later, separate commands stopped the same policy at pick/hold and resumed it for tray placement. | “Pick” needed to mean hold, and the user wanted left/right hands, tray unloading and other destinations. | Four separate pick/hold/place cycles passed on seed 210000. Broader development results: **11/12 single picks, only 2/8 complete scenes** with the 30-action setting. Right-hand table-to-tray scope; not a general household skill. |
| **Independent ACT primitives** — feat/r1pro-primitives-integration; earlier version in feat/r1pro-act-sim | Four models: right pick, right place, left pick, left place. Each controls only that arm and gripper. Classical reachability and positioning precede ACT. | Bench results did not transfer reliably to the apartment or the actual interactive starting states. | Working interfaces and some physical successes, but unreliable policies. Latest corrective round: **3/8 complete sequences**, **3/7 integrated cases**, no improvement over its 3/8 baseline. First-generation preview remains selected; newer experiments were not promoted. |
| **Apartment ACT** — feat/r1pro-primitives-integration, also contained in classical branch | Randomized procedural props across worktable, kitchen and dining table; body/torso positioning, navigation, then independent ACT pick/place. | Repeated failed grasps persisted after staging fixes. User explicitly chose a separate classical path to validate locomomanipulation. | Appearance refinement: **0/4 apartment cases**. Reach refinement: **0/8**. A real arm-staging bug was fixed, but subsequent ACT picks still failed. **No policy-apartment-preview bundle was promoted; it is absent.** |
| **Classical apartment** — feat/r1pro-classical-apartment | GraspGenX proposes grasps; DimOS chooses arm/body pose, approaches, grasps, carries, places and verifies physical results. Independent HumanCLI skills. No ACT runs. | Narrow routes, docking failures and slow execution made debugging difficult. User requested an open test area. | Successful six-action delivery/re-pick sequence; reproduced cabinet collision fixed and **seven-action regression passed**; cross-hand and two-held-object **eight-action regression passed**. User also had successful HumanCLI picks. These are measured scenarios, not broad reliability guarantees. |
| **Classical open space** — same classical branch | Same manipulation pipeline on a 12 × 12 m floor with five named platforms at different heights. Removes apartment walls/cabinets from the navigation problem. | Latest work focused on severe simulation slowdown rather than adding capabilities. | **Ten-action seed-5000 sequence passed**, including re-pick and two destinations; separate desktop startup passed. Latest desktop slowdown remains unresolved overall: viewer blocking fixed in an uncommitted change, but full delivery acceptance of that change is incomplete. Host active-core clocks later measured around **200 MHz**. Paused for reboot. |

The R1Pro planar-base preview on **feat/r1pro-mobile-manip** is supporting planning work, not another finished pick-and-place demo. Its r1pro-planar-preview blueprint uses fake hardware to preview base/upper-body planning. It should not be confused with the physical-contact MuJoCo demos.

### What is genuinely reusable

- **Constrained learned behavior:** the five-bottle ACT checkpoint and the full tray-delivery composition.
- **Interactive execution structure:** separate requests, selected-object/hand checks, action status, stop/recovery, and explicit reset.
- **Classical manipulation integration:** GraspGenX proposals, DimOS IK/posture planning, measured TCP alignment, physical grip/support checks, and preservation of the other hand’s object.
- **Navigation integration:** full-scene point cloud into KronkNav, DimOS holonomic execution, loaded-geometry checking and measured arrival compensation.
- **Learning infrastructure:** demonstrations, conversion, checkpoint migration, preserved normalization, incremental training, and physical evaluation. This infrastructure working does not mean the resulting policy succeeds.

### What is not finished

Reliable ACT grasps and placement across random objects, hands, heights and apartment locations remain unfinished. The requested short ACT skill that starts at a classically obtained pregrasp was discussed; a new 100–200-demonstration contact-only pilot was **not completed** before switching to the classical branch.

We also do not have realistic arbitrary household assets or real-camera perception in these demos. The “cup” is a narrow hollow handleless cylinder; cartons, glue sticks and toys are procedural approximations. The latest classical branch does not automatically include the older tray-carrying demo’s bed/floor/tray-delivery features.

For an ACT demonstration, retain the **five-bottle home demo** as the established narrow baseline. For continued interactive locomomanipulation work, retain the **classical branch**, beginning with a known tested open-space seed after the host is healthy. Treat primitive/apartment ACT as research work.

## LLM continuation context

### Read this first

The last user action was to **pause execution**, followed by this documentation request. Do not automatically restart training, a simulator, or a hardware stack after reading this file. Ask the new conversation’s user which line of work they want to resume if they have not specified it.

The desired product is an interactive robot: pick a named/spatially described item with a requested hand; hold it; navigate on a separate request; place it in a named free supported region; preserve the other hand’s object. “Pick” must not imply “put in tray.” Explicit hand/target selections must not be silently changed. Reset must be explicit.

The user originally required ACT for both pick and place, but later explicitly authorized **a separate classical GraspGen-based branch**. That later instruction is the applicable scope for the classical demo. Do not rewrite it back into ACT or delete the ACT experiments.

Use full desktop MuJoCo display and Zenoh for user demos. The user asked for approximately 30 Hz display updates and simple launch commands. Headless/EGL evaluation is a separate development mode. Long training jobs, if resumed, must survive terminal disconnects and be checked periodically. No new training is needed to run the classical demo.

### Actual directories and Git state

Remote status below is relative to the locally cached origin refs inspected in this audit; no fetch or push was performed. Full hashes and machine-readable details are in [inventory.json](inventory.json).

| Branch / pointer | Tip | Location / role | Preservation and remote status |
| --- | --- | --- | --- |
| hackmit/t7-resume | 0b816edf49 | /home/mustafa/dimos-wt/hub | Completed OpenYAM source. No configured upstream; no cached remote branch contains this exact tip. |
| mustafa/task/hackmit-manip | 0b816edf49 | Alias pointer to same OpenYAM tip | Not a separate implementation. |
| feat/openyam-sim-completion | fd94f07ece | Earlier completion checkpoint | Ancestor of hub; predates final T7 training/control/agent fixes. |
| feat/r1pro-act-sim | 0187ef1208 | /home/mustafa/dimos-wt/r1pro-act-sim | **42 total commits ahead** of origin/feat/r1pro-act-sim at 389632263b. This count includes merged upstream changes, not 42 distinct policy features. Tracked files clean at audit. |
| feat/r1pro-primitives-integration | b698b416cb | Historical development pointer; no separate current worktree | Includes apartment ACT/staging fixes. Its exact tip is an ancestor of the classical branch and is contained in cached origin/feat/r1pro-classical-apartment. |
| feat/r1pro-classical-apartment | ebd067decc | /home/mustafa/dimos-wt/r1pro-classical-apartment | Committed tip matches its cached remote. **Eight changed/new project files remain uncommitted**, listed below. |
| feat/r1pro-mobile-manip | 8f2138cfb9 | Supporting planar planning preview | Its upstream is marked gone. Not an ACT task policy or a validated apartment delivery. |
| backup/r1pro-before-home-blueprint-rebase | bb39ed4d3f | Recovery checkpoint before home blueprint rebase | Preserves initial loaded-tray KronkNav integration; not a preferred new starting point. |
| backup/r1pro-before-random-objects-20260911 | 14f90296c6 | Recovery checkpoint around interactive bottle demo | Useful historical reference, not a separately maintained demo. |
| chore/r1pro-before-apartment-main-rebase-20260913 | 145ffe5147 | Pre-rebase apartment development snapshot | Recovery pointer; rebased successor work is in primitive/classical history. |

The old hackmit/t1 through hackmit/t5b worktrees are foundation milestones: generic MuJoCo whole-body binding; multiple cameras; privileged perception; per-arm grippers; classical demo; left-arm/retry fixes. Their commits are preserved. Continue OpenYAM from hub rather than treating each as an independent current demo.

This is **not a simple linear Git ancestry diagram**. Work was rebased and integrated across branches. For example, primitive integration is an ancestor of classical, but the exact current ACT tip is not. Do not infer that all work was lost or all files differ from that fact alone; compare source/patches before moving changes.

The root checkout /home/mustafa/dimos is on **krishna/feat/alfred**, with unrelated work. It is not the current R1Pro demo checkout. Do not switch/reset it to fix a blueprint lookup. A branch already checked out in a worktree should be used from that directory.

Main is not continuously synchronized now: ACT’s merge base with cached origin/main is 9314c76543; classical’s is aa8a158469 (September 13 integration). Cached origin/main is dd2b00ac2a at audit. Historical handoff claims of “current main” apply to their dates. A future rebase/merge should be deliberate, after preserving the dirty classical changes.

### Runtime and artifact locations

| Item | Canonical local path / dependency |
| --- | --- |
| OpenYAM source | /home/mustafa/dimos-wt/hub |
| OpenYAM recording, dataset, trained checkpoints and evidence | /home/mustafa/dimos/recordings/openyam-completion |
| R1Pro ACT source | /home/mustafa/dimos-wt/r1pro-act-sim |
| R1Pro classical/current apartment source | /home/mustafa/dimos-wt/r1pro-classical-apartment |
| Shared R1Pro recordings root | /home/mustafa/dimos-wt/r1pro-act-sim/recordings |
| R1Pro learned artifacts and jobs | Shared recordings root / r1pro-act-task |
| House scene package | /home/mustafa/dimos/data/scene_packages/hssd_102344115 |
| ACT runtime | ACT worktree / .home-venv |
| Classical runtime | Classical worktree / .classical-venv |

The classical recordings directory is a **symlink to the ACT recordings root**. Its .home-venv also points to ACT’s .home-venv. Both worktrees have .venv links to the root environment. The isolated runtimes still reuse some packages from the root environment; they are not entirely independent installations.

The .home-venv editable path points at ACT source. The .classical-venv path configuration places classical source first. Merely changing directories while using a different checkout’s installed dimos command can produce “Unknown blueprint” or run older code.

Datasets, checkpoints, scene assets, runtime environments and logs are local artifacts, not part of the ordinary source commits. A Git push does **not** back them up. Preserve the source recording, converted dataset metadata, policy weights, processors/normalization files, deployment.json and scene/model manifests together. Do not prune these to recover disk space without reviewing what is actually disposable.

### Blueprint map and behavior

The source filenames below are under dimos/robot/galaxea/r1pro unless an OpenYAM path is given.

| Blueprint / entry point | Source | Starts idle? / purpose | Policy selection and important limits |
| --- | --- | --- | --- |
| dual-openyam-sim-agent | manipulators/dual_openyam/blueprints/agentic.py | Idle classical agent | Two stationary arms; simulation perception. |
| dual-openyam-sim-policy-agent | Same OpenYAM agentic source | Exposes policy controls as well | OpenYAM ACT bonus failed physical success benchmark; do not present as trained reliable behavior. |
| demo_pick_place_stack Python module | demo_pick_place_stack.py | Automated single-bottle test/delivery | policy-house is old fixed-tray baseline; policy-free-tray is the physical handled-tray adaptation. |
| r1pro-home-sim | home_blueprint.py | **Automatically** packs five bottles and delivers tray | policy-packing-augmented; no LLM needed. |
| r1pro-home-sim-agent | home_agent_blueprint.py | Idle interactive bottle/tray agent | Same narrow bottle ACT; classical tray handling/unloading; selected bed/floor patches belong to this older demo. |
| r1pro-objects-sim / -agent | object_agent_blueprint.py | Idle selected-object skills; agent adds language | policy-objects-interactive. Right-hand table pick and tray place using one split full-sequence model. No general room delivery. |
| r1pro-primitives-sim / -agent | primitive_blueprint.py | Idle four-policy primitive preview | policy-primitives-preview/{pick-right,place-right,pick-left,place-left}; experimental. |
| r1pro-apartment-sim / -agent | apartment_blueprint.py | Intended apartment ACT extension | Requires policy-apartment-preview, which is absent/unpromoted. **Do not give this as a working default launch.** |
| r1pro-classical-apartment-sim / -agent | classical_blueprint.py | Idle independent classical skills; agent adds language | No ACT. GraspGenX + DimOS + physical MuJoCo checks. |
| r1pro-classical-open-space-sim / -agent | open_space_blueprint.py | Same skills on five platforms | No ACT. Same classical branch, not a new branch. |

The R1Pro non-agent interactive variants still expose MCP for explicit skill calls. The automatic home blueprint is a separate behavior; a “-sim” suffix alone does not universally mean idle.

**Reference launches, for when the user resumes:**

Established five-bottle ACT demo:
~~~bash
cd /home/mustafa/dimos-wt/r1pro-act-sim
.home-venv/bin/dimos run r1pro-home-sim
~~~

Interactive bottle/tray version:
~~~bash
cd /home/mustafa/dimos-wt/r1pro-act-sim
.home-venv/bin/dimos run r1pro-home-sim-agent
# Second terminal, same directory:
.home-venv/bin/dimos humancli
~~~

Classical open space:
~~~bash
cd /home/mustafa/dimos-wt/r1pro-classical-apartment
.classical-venv/bin/dimos run r1pro-classical-open-space-sim-agent
# Second terminal, same directory:
.classical-venv/bin/dimos humancli
~~~

Classical apartment uses the same runtime with r1pro-classical-apartment-sim-agent. ACT bench preview uses ACT’s .home-venv with r1pro-primitives-sim-agent and remains experimental. These commands identify the correct sources; they were not launched during this inventory.

The agent variants need the user’s configured language-provider credentials in the **launching** process. HumanCLI does not supply a missing key to an already running agent. Local MCP and non-agent demos need no external language inference. Prior user-driven live success is not an automated broad language benchmark.

The legacy OpenYAM launcher still needs its documented source/display setup because hub’s .venv points at the root environment. Use hub/docs/demos/openyam-simulation.md rather than copying R1Pro launch assumptions into it.

### How ACT is used here

ACT predicts chunks of joint-position commands from observations. DimOS supplies observations, runs the policy in the isolated LeRobot runtime, submits the action chunk to ControlCoordinator, and checks the physical outcome. ACT does not choose a room, navigate, solve tray packing, or create a semantic scene model by itself.

| Generation | What the model receives / controls |
| --- | --- |
| Single-bottle R1Pro | Head/right-wrist RGB and measured robot joints; learns the constrained pick/deposit sequence. |
| Five-bottle packing | RGB, 20 measured joint positions, and eight goal values: source XYZ, destination XYZ, radius and half-height. Outputs the bottle grasp/place/release/home joint sequence. Geometry selects bottle/slot outside ACT. |
| Random-object packing | RGB and joints plus 52 geometry features: selected object and destination relative to TCP, object rotation, half-sizes, one-hot cylinder/box/bottle type, home offset, and four neighbors’ presence/relative positions/extents. This is explicit geometry conditioning, not just task text. |
| Independent primitives | Head and active-wrist RGB, the active hand’s eight joint/gripper values, relevant geometry and other-arm/torso context. Pick drops destination XYZ features. Each model outputs only seven arm joints plus its gripper; base and torso are positioned separately. |

These models were not merely looking up an object ID or trained only on one exact position. However, conditioning inputs and randomized data did **not** establish reliable transfer. Narrow training distributions, interactive starting-posture differences, approach-centering errors, destination error, rollout settings and actual implementation bugs all mattered.

Matched traces found failed ACT approaches roughly 1–2 cm off-center before closure; the classical teacher passed those same states. One ACT placement missed its requested region by about 5 cm. A separate apartment bug skipped the arm-ready motion whenever the torso did not need to move; this was fixed by checking all required joints. Both hands still failed ACT tests after correct staging. Another diagnostic exposed restrictive policy action clipping, but changing only that clipping did not fix the grasps.

**Training accounting:** “30k” referred to optimizer updates, not 30,000 new demonstrations.

- OpenYAM: 100 episodes, 33,319 aligned frames; a 30,000-update ACT run and 10k/20k/30k evaluation.
- R1Pro first task: 60 tabletop demonstrations; task model trained for 4,000 updates, then house/free-tray adaptations from existing weights.
- Five-bottle packing: 130 unique picks, split 120 training / 10 held-out; later continuation and image augmentation improved the narrow task.
- Random-object pilot: 115 demonstrations, 10,000-update pilot followed by 30,000 additional updates on the same data.
- Primitive work: segmented old demonstrations, added both-hand/source/destination/carry contexts, migrated compatible weights and normalization, then fine-tuned separate artifacts.
- Later corrections preserved rehearsal data and old checkpoints. They were not all fresh training from scratch. A completed optimizer run was never sufficient for promotion.

### Which policy files are actually selected

Paths are relative to shared recordings/r1pro-act-task.

| Artifact | Actual role / audit finding |
| --- | --- |
| policy | Selected original tabletop checkpoint. |
| policy-house | House adaptation for old fixed-tray task. |
| policy-free-tray | 1,000-update free-tray adaptation; single-bottle profile. |
| policy-packing-augmented | Five-bottle default, exported from train-packing-augmented/checkpoints/005000/pretrained_model; 30 executed actions per observation. |
| policy-objects-interactive | Random-object default, exported from jobs/random-objects-act-refine-v1/policy; 30 executed actions. |
| policy-primitives-preview | preview.json points to **jobs/independent-primitives-act-v1/policies**, the first independent-policy generation, with 30-action deployment. Later refinements did not replace it. |
| policy-apartment-preview | **Does not exist** at audit. Registered blueprint is not equivalent to a runnable validated bundle. |
| policy-packing-flexible-* and other intermediate packing exports | Experiments, not the selected augmented default. Do not choose by largest step count or newest filename. |

The free-tray, five-bottle, random-object and eight-output primitive profiles have different input/action contracts. Swapping artifact directories without checking profiles, joint ordering and normalization is invalid.

### Latest completed experiment status

These values were read from existing job JSON, not inferred from old PIDs.

| Job | Saved result | Interpretation |
| --- | --- | --- |
| random-objects-act-v1 | 6/12 singles, 0/8 scenes | Initial learned pilot. |
| random-objects-act-refine-v1 | 9/12 singles, 1/8 scenes | Saved 20-action setting. Later unchanged-weight 30-action evaluation improved to 11/12 and 2/8. |
| independent-primitives-act-v1 | Completed; 1/8 at its initial evaluation setting | Different 30-action comparisons reached 7/8 on a small bench set; not universal success. |
| primitive-place-refine-v1 | Completed; 4/8 in saved job result | Later matched routing-corrected bench comparison gave 7/8 on different seeds. Keep the protocols distinct. |
| primitive-interactive-refine-v1 | Completed; 3/8 in saved job result | Expanded interactive context did not establish reliable behavior. |
| primitive-approach-refine-v1 | Completed; old 3/8, new 3/8; integrated 3/7 | No measured overall improvement; not promoted. |
| apartment-appearance-refine-v1 | Completed; 0/4 apartment cases | Not promoted. |
| apartment-reach-refine-v1 | Completed; 0/8 apartment cases | Mix of policy and positioning/navigation failures; not promoted. |

Do not compare counts across different seeds/protocols as a clean learning curve. Teacher successes, offline ACT results, native MCP results, and user-driven HumanCLI results are separate evidence categories.

### Classical execution: source and responsibilities

The classical path is a module-driven action state machine. HumanCLI → McpClient → MCP skills → geometry/planning → ControlCoordinator → simulated actuators → measured outcome. GraspGenX is itself a pretrained learned grasp proposer, so “classical” here means **no ACT motor policy**, not “no neural model anywhere.”

| Source under dimos/robot/galaxea/r1pro | Responsibility |
| --- | --- |
| classical_skills.py | Independent pick/go/place actions, phases, cancellation/recovery, action evidence. |
| classical_selection.py | Resolve requested IDs, colors/types and robot-relative side/distance; reject mismatches/ambiguity. |
| classical_perception.py | Selected-object segmented point cloud using simulation geometry and virtual depth views. |
| classical_sim.py | Snapshot/RPC planning boundary, live physical state reporting, carry/navigation checks. |
| classical_planning.py | Rank GraspGenX candidates and whole-body choices; check approach, closure, lift, transfer and placement. |
| home_kinematics.py | DimOS manipulation SDK model and Pink IK; uses DimOS’s RoboPlan-backed world. “Native” refers to compiled backend code, not an alternative application outside DimOS. |
| object_reachability.py / primitive_workspace.py | Physical reach, candidate body poses and policy/workspace distinction. |
| primitive_scene.py / placement_regions.py | Supported placement regions, footprint/clearance and source/goal state. |
| classical_gripper.py / object_primitive_state.py | Gripper/hold/support behavior and per-hand object ownership. |
| apartment_navigation.py / apartment_route.py | KronkNav request, corridor/departure/arrival handling and loaded motion integration. |
| primitive_coordinator.py / navigation_base.py | Task composition and idealized planar base servo. |
| open_space_scene.py / open_space_sim.py | Open floor and worktable, low_bench, display_table, tall_table, high_counter. |
| primitive_skills.py / primitive_policies.py / object_primitives.py | The separate ACT primitive path; do not mistake these for the classical motor controller. |

For a classical pick: resolve target → acquire segmented geometry → propose grasps → rank free/requested hand and body/torso pose → collision-check → navigate/preposition → measured TCP alignment → approach → close → verify both pads → lift/hold.

For a place: retain measured grasp → find a free supported footprint and feasible body pose → carry/align → descend with measured compensation → verify intended support → open → retreat. It must preserve the other hand’s object.

The grasp planner also tries body-centered translations for the symmetric procedural props while keeping the proposed grasp orientation. This is an explicit heuristic refinement, not a claim that every executed pose is an untouched GraspGenX output.

### What the simulation gives us for free

- Exact instance identity, color/type, object pose/dimensions and physical geometry.
- Virtual multiview depth centered on the selected object, rather than proven real-camera detection/segmentation.
- A full static environment point cloud and idealized localization.
- Exact contact forces/object motion for grip, support, spill and slip checks.
- A planar actuated mobile base, rather than a validated wheel-ground hardware dynamics model.

Current physical-tray and classical manipulation do not weld or teleport held objects. Attachments used for collision planning are confined to copied planning scenes. Episode reset/training initialization sets up scenes. The earliest floating/fixed-tray prototype is the historical exception and was replaced for real tray delivery.

### Evidence worth keeping

R1Pro paths below are relative to the shared recordings directory unless stated otherwise.

| Evidence | What it establishes |
| --- | --- |
| /home/mustafa/dimos/recordings/openyam-completion/takeover-live-verification.json | User-driven live OpenYAM completion independently checked for containment and home joints. |
| OpenYAM dataset-100 and policy-eval-10000/20000/30000 reports | Data/training completed; stable deposit failed all checkpoint tests. See hub/docs/demos/openyam-validation.md for exact report names. |
| r1pro-act-task/home-zenoh-5000/result.json | Complete five-bottle home run. |
| r1pro-act-task/jobs/home-zenoh-5000/validation-summary.json | 19.67 s navigation, 0.6004 m/s peak, continuous bimanual grasp, upright cargo, tray error 4.22 mm. |
| r1pro-classical-navigation-fix/seed-1646757217/result.json | Seven successful actions after reproducing cabinet collision. |
| r1pro-classical-navigation-fix/cross-hand-1498950867/result.json | Eight successful actions including cross-body left pick, both hands holding during travel, separate tray placements. |
| r1pro-classical-open-space/seed-5000/validation.json | Ten successful commands including two destinations, re-pick, and left cup handling. |
| r1pro-classical-open-space/viewer-stall-20260917/ | Latest performance investigation, passing/failed runs, thermal observations and viewer isolation checks. |

The ten-action open-space sequence was headless physical validation; a separate seed-5001 desktop startup was checked. Do not relabel it a full ten-action desktop acceptance. Historical scenario success does not certify the latest dirty source at all seeds.

### Current uncommitted change and exact stopping point

Classical branch tip **ebd067decc** is pushed according to cached origin. It contains the cup-contact optimization and earlier 15 Hz viewer setting. **eb0a524eb3** is the earlier open-space feature checkpoint.

The dirty worktree adds process-isolated display and 30 Hz state publication:

- dimos/simulation/engines/mujoco_viewer.py — new bounded snapshot transport and separately owned desktop viewer process.
- dimos/simulation/engines/mujoco_engine.py — viewer publisher/lifecycle; physical simulation does not wait for native GUI sync.
- dimos/simulation/engines/mujoco_sim_module.py — configuration plumbing.
- dimos/robot/galaxea/r1pro/classical_blueprint.py — enables separate display and 30 Hz updates.
- dimos/simulation/engines/test_mujoco_viewer.py — new IPC/frame behavior tests.
- dimos/simulation/engines/test_mujoco_engine_timing.py — physics progress/lifecycle regressions.
- dimos/robot/galaxea/r1pro/OPEN_SPACE.md — launch/performance limits.
- openspec/changes/r1pro-act-house-sim/handoffs-open-space.md — exact investigation handoff.

Pre-existing untracked environment/data links are not project source additions. Do not stage them with a blanket git add.

Measured facts:

1. Native inline viewer sync stalled for **12.42 s while holding the physics lock**.
2. After process separation, deliberately suspending only the viewer for 12 s left physics at **1.00031× real time**, with scene RPC calls around 10–40 ms.
3. All **37 focused viewer/engine/module tests passed** after final lifecycle/pacing edits. Mypy/pre-commit passed before those last small edits; rerun them before committing.
4. Full latest-seed delivery is **not accepted**. Navigation passed, but a glue-stick closure pushed the object over; physical grasp verification failed. Recovery properly refused the unstable state.
5. A later diagnostic lost its simulation worker. Kernel crash address was Python’s PyUnstable_InterpreterFrame_GetLine and periodic faulthandler output ended mid-frame. Diagnostic stack dumping is the leading suspect, not a proven general simulator crash cause. It was removed for the next run.
6. That repeat remained slow and was stopped at user request. The final host sample showed a busy worker using 98.2% of a core while that active core read about **200 MHz**; other active cores also read about 200 MHz. User reported roughly 90°C even near 2% total utilization, with clear vents and fans moving air.
7. Earlier 400–800 MHz samples included idle cores and were overinterpreted. Later active-core 1.6–2.5 GHz samples had raw MuJoCo step cost 0.40–0.63 ms. The final 200 MHz observation is distinct. We have not diagnosed the specific cooling/power fault.
8. All owned diagnostic simulation/test processes were stopped. Code and copied evidence were retained. No power/fan settings were changed. No further work should resume merely because an old status file lists a PID.

The process viewer is display-only for native physics edits: camera orbit/zoom work, but editing physics through the viewer affects its private copy. Use DimOS commands for robot state. Native UI redraw can exceed the 30 Hz snapshot rate.

Repeated HomeKinematics construction was identified as potentially expensive. **No planner-model cache was implemented.** Avoid presenting that proposed optimization as a completed change.

### How to resume without repeating this history

1. Agree on one target: constrained ACT tray demonstration, classical mobile manipulation, or a research experiment on short ACT contact skills.
2. Preserve the dirty classical changes and local data. Establish host temperature/clocks after reboot before interpreting simulation speed or starting a long run.
3. Select the matching checkout, runtime, blueprint and policy contract. Check imported dimos source if a blueprint is unknown.
4. Reproduce a recorded successful seed first using local MCP and saved physical outcomes. For classical open space, seed 5000 is the baseline; apartment regressions include 1646757217 and 1498950867.
5. Change one source of failure at a time. Keep actual grip/support/collision checks, target/hand intent and loaded-object preservation. An accepted command or completed trajectory is not manipulation success.
6. Only after a full post-change run should the viewer patch be treated as accepted. Then commit/push deliberately without Codex coauthor trailers, as the user requested earlier.
7. For ACT research, retain existing weights/data. Start a bounded, measured comparison at verified pregrasp states with held-out positions/objects; do not schedule another large unchanged-data fit just because earlier fitting completed.

This inventory itself is documentation-only and uncommitted. It does not claim that the broader interactive random-everyday-object product is complete.

### Deeper source documents

Use the documents in the checkout relevant to the chosen branch. Many contain chronological sections with superseded statuses.

- OpenYAM: hub/docs/demos/openyam-simulation.md; hub/docs/demos/openyam-validation.md; hub/openspec/changes/hackmit-dual-openyam-sim-demo/handoffs.md.
- R1Pro single bottle/tray: ACT_SIM.md; handoffs-tray-baseline.md.
- R1Pro five-bottle home: BOTTLE_PACKING.md; handoffs.md.
- Transitional random objects: OBJECT_INTERACTIVE.md.
- Independent ACT: OBJECT_PRIMITIVES.md; handoffs-random-objects.md.
- Apartment ACT: APARTMENT_SIM.md and the **final** apartment deployment failure section of handoffs-random-objects.md in the classical/integration source.
- Classical apartment: CLASSICAL_APARTMENT.md; handoffs-classical-apartment.md.
- Classical open space/current pause: OPEN_SPACE.md; handoffs-open-space.md.

All R1Pro product documents are under dimos/robot/galaxea/r1pro. R1Pro handoffs are under openspec/changes/r1pro-act-house-sim. Prefer the classical checkout for the latest apartment/classical history: the ACT checkout’s handoff stops earlier.
