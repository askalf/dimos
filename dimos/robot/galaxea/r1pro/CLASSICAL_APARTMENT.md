# Classical R1Pro apartment

A separate interactive demo on `feat/r1pro-classical-apartment`. The robot uses GraspGenX and DimOS planning; no ACT policy runs in this stack. Transport and varied-layout validation are still in progress. See the [handoff](../../../../openspec/changes/r1pro-act-house-sim/handoffs-classical-apartment.md) for measured results and remaining failures.

## This workstation

Close the previous demo with Ctrl-C before starting another stack on the default MCP port. The isolated runtime has already been installed; no environment exports are needed:

```bash
cd /tmp/dimos-r1pro-primitives
.classical-venv/bin/dimos run r1pro-classical-apartment-sim-agent
```

In a second terminal:

```bash
cd /tmp/dimos-r1pro-primitives
.classical-venv/bin/dimos humancli
```

The agent uses your existing configured provider credentials. The full MuJoCo viewer opens. A new launch randomizes the seed, object properties and positions across the worktable, kitchen and dining table. `reset_scene` repeats the current seed. Pick and place are independent commands; the robot starts idle.

Examples, using the colors and IDs present in the current scene:

- “What objects are near you?”
- “Pick up the light blue glue stick on your left with your left hand.”
- “Pick object_2 with your right hand.”
- “Go to the dining table.”
- “Place the object in your left hand on the dining table.”
- “Stop.”

An absent or ambiguous description is rejected before movement. An occupied hand is rejected. A pick ends with the object held; navigation preserves holds; only a place command releases an object. Failed actions expose their phase and evidence. Recovery preserves a confirmed hold or restores an empty hand when the scene checks permit it; it never resets progress or retries a grasp silently.

For local skill calls without a language model, start `r1pro-classical-apartment-sim` instead, then use:

```bash
.classical-venv/bin/dimos mcp call get_scene
.classical-venv/bin/dimos mcp call pick_object --json-args '{"object":"nearest","arm":"auto"}'
.classical-venv/bin/dimos mcp call wait_for_action --arg seconds=20
```

Repeat `wait_for_action` until its state is terminal. Inspect `get_surfaces` before choosing a placement region. `go_to` accepts `worktable`, `dining_table`, or `kitchen`.

## Execution

1. Resolve every requested color/type/side qualifier against the current scene.
2. Raycast the selected object into a world-frame segmented point cloud.
3. Generate GraspGenX candidates for the measured R1Pro parallel gripper. Also test translations centered on the body of these symmetric props, retaining the generated grasp orientation.
4. Compare free hands and base/torso postures using DimOS IK, joint margins and manipulability. Check approach, jaw closure, lift and apartment collisions.
5. Navigate if necessary, position the base, execute a collision-checked SDK posture path, and correct the measured TCP pose before approaching.
6. Close, verify force-backed contact on both pads, lift and verify the hold.
7. On a separate place request, intersect empty support space with a feasible body/arm posture; descend, confirm the requested support, open and retreat.

KronkNav consumes the full apartment point cloud. Its route must also pass a check of the robot's complete swept geometry, including the held objects and wrist cameras. The DimOS holonomic task executes the accepted path. A compact carrying posture folds both arms before travel. Loaded placement transfers use Cartesian IK to preserve the object attitude and the other hand, with full collision checks. Base limits are 0.3 m/s and 0.4 rad/s; loaded torso changes use lower joint speed and acceleration to retain the grasp.

Simulation instance labels, object geometry and virtual multiview depth viewpoints are used deliberately. This validates planning/control integration, not real-camera semantic perception. The current assets are procedural bottles, cups, cartons, glue sticks and toy blocks. Arbitrary household meshes, floor/bed placement, tray transport and hand-to-hand transfers are outside this branch's tested scope.

All object attachments are confined to planning copies. Live objects are held and supported through MuJoCo contact physics.

## Development validation

The local MCP harness runs without an external language model:

```bash
.classical-venv/bin/python -m dimos.robot.galaxea.r1pro.demo_classical_apartment \
  --output /tmp/my-classical-test \
  --seed 282527379 \
  --mcp-port 10026 --zenoh-scout-addr 224.0.0.224:19492 \
  --actions pick:right:object_2 place:right:worktable \
  --viewer --stay-open
```

Use a fresh output directory and unused test ports. Per-action JSON contains requested/resolved targets, planning choices, outcome and physical state; `result.json` also records source hashes. Workstation background validation status is under `/tmp/r1pro-classical-bootstrap/*-status.json`.

For a fresh environment the project provides `manipulation` and `graspgenx` extras. GraspGenX source and model revisions are pinned by the existing DimOS provider. The workstation's `.classical-venv` adds that provider to the existing apartment runtime without modifying the user's ACT environment.
