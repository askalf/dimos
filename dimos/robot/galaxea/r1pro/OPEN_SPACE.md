# R1Pro open-space demo

An interactive classical pick, carry and place demo on a 12 × 12 m floor, with five named platforms and no apartment walls or cabinets. The platforms retain physical collision geometry. GraspGenX proposes grasps; DimOS handles reachability, body positioning, navigation and manipulation. No ACT policy runs.

## Run

Close the previous demo with Ctrl-C, then:

```bash
cd /home/mustafa/dimos-wt/r1pro-classical-apartment
.classical-venv/bin/dimos run r1pro-classical-open-space-sim-agent
```

In another terminal:

```bash
cd /home/mustafa/dimos-wt/r1pro-classical-apartment
.classical-venv/bin/dimos humancli
```

The full MuJoCo viewer and Zenoh are defaults. The agent uses the existing provider configuration. No environment exports are needed. Scroll to zoom in on manipulation or labels. The robot starts idle.

| Platform | Height | Color |
| --- | --- | --- |
| `worktable` | 70 cm | Blue |
| `low_bench` | 60 cm | Green |
| `display_table` | 80 cm | Purple |
| `tall_table` | 90 cm | Orange |
| `high_counter` | 85 cm | Red |

A tray remains on the worktable. Each platform starts with one object: the existing cup, bottle, drink carton, glue stick and toy block, with randomized colors, dimensions, assignments and supported positions. Each new launch randomizes the seed; `reset_scene` repeats the current seed.

## Interact

Ask what objects are present before selecting one. For example:

- “Pick the carton with your right hand.”
- “Go to the low bench.”
- “Place the object in your right hand on the low bench.”
- “Pick that carton up again.”
- “Take it to the high counter and place it there.”
- “Pick the cup with your left hand.”
- “Place it on the display table.”

A pick ends with the object held. Navigation preserves holds. Placement verifies support before opening the gripper. A requested hand is preserved; the other hand may keep holding an item. Use `get_surfaces` for exact region names. The room names from the apartment do not apply here.

The tray is a physical free body. `place_object` with region `tray` puts a held item into it wherever it currently rests. `pick_up_tray` docks in front of the resting tray and lifts it with both hands, keeping its contents; both hands must be empty. `go_to` carries a held tray and docks where it would be set down. `put_down_tray` carries the tray to a named platform, lowers it onto a clear footprint near the robot's edge, verifies support, releases and retreats. Nothing can be picked or placed while the tray is held. The tray planner measures the handle spacing from the model; the classical tray is wider than the home demo's, and the older fixed constant closed the fingers 4 cm inboard of the handles.

The tray is carried with its bottom about 85 cm above the floor, and the torso is already at its tallest level posture, so platforms at or above that height (tall_table, high_counter) cannot receive the tray. `put_down_tray` and a loaded `go_to` refuse them up front with the measured heights. The dock beside a platform steps back along the approach heading until the robot and its carried tray are clear of the platform legs; the arms cover the remaining distance. The footprint search also avoids fixtures standing on the platform. In the packaged apartment this leaves only the worktable: the laptop, lamp, journal and camera on the dining table leave no clear 32 by 47 cm footprint for this tray, and the kitchen counter is above the carry height.

This scene uses the same privileged simulation perception and contact checks as the [classical apartment](CLASSICAL_APARTMENT.md). It simplifies navigation geometry; it does not establish arbitrary-object or hardware reliability.

## Verified run

Seed 5000 completed ten consecutive commands through local MCP: right-hand carton pick, low-bench delivery and re-pick, high-counter delivery, navigation to the display table, left-hand cup pick, and delivery to the tall table. Both final objects were physically supported, released, upright and settled. A separate full GLFW viewer startup passed at seed 5001. These checks did not call the language model.

Evidence is saved locally in `recordings/r1pro-classical-open-space/seed-5000/validation.json`, with per-action states and a scene preview in the same directory. The focused suite passed 68 tests; strict type checking passed on ten changed production sources.

## Simulation speed

The desktop viewer receives state updates at 30 Hz; physics retains its 2 ms timestep (500 steps per simulated second), and camera streaming remains on a separate thread at 10 Hz. The shared simulator exposes `viewer_fps` separately from camera `fps`. The classical demo now sends bounded snapshots to a separate viewer process. Slow viewer synchronization drops display frames instead of holding the physics lock. Camera orbit and zoom remain available; use DimOS commands to modify robot state, since native viewer physics edits affect only the display copy. The native UI may redraw faster than the 30 Hz state updates.

The object labeled `cup` is a narrow, hollow, handleless cylinder, not a detailed mug asset. In the profiled open-space run (seed735730399), it was the orange `object_5` on `display_table`; IDs and positions change across seeds. Its walls now sit on its own bottom disk. Previously the disk and all 24 wall segments touched the table, producing 101 cup/table contacts at rest. The corrected geometry produces five, while preserving the hollow cavity, physical grasp surfaces, outer dimensions and total mass. On the saved slow-run scene this reduced raw physics cost from 2.93 ms to 0.34 ms per step. This is a physics benchmark, not an end-to-end action latency guarantee: GraspGenX and reachability planning still take time.

Restart the blueprint to generate the corrected scene; `reset_scene` reuses the existing model. No new environment exports are needed.

### Performance status

The September 17 desktop run exposed a 12.4-second viewer synchronization stall while holding the live physics lock. The snapshot viewer removes that coupling: a real 12-second suspension of only the viewer left physics running at approximately real time, and its 37 focused simulation/IPC tests pass.

After the host reboot the same day, active cores idled at 0.8 to 1.6 GHz with package temperatures near 60 to 70 C, and the seed 5000 three-action sequence (right carton pick, low-bench delivery and placement) passed headless on the snapshot-viewer code in 6.5 minutes of wall time. The earlier 200 MHz readings were a thermal or power fault of the host, not a simulation regression; check clocks and temperatures again before interpreting any slow run.

## Grasp assessment

Ranking GraspGenX proposals runs in a separate worker process that the simulator starts and warms at the first session call, so the kinematics world (about a minute to build) is paid once, before the first pick. Each pick then ranks in roughly 20 to 40 seconds; the skill polls with a cancellable wait and gives up after 150 seconds with a plain reason ("free the other hand or ask again from closer"), killing the worker so a runaway search cannot starve physics. The worker logs to `assessment-worker.log` in the session directory; each request lives in a `classical-assessment-*` directory with its `result.json`. When the free hand is across the body from the target, the ranker tries repositioned stances first.

## Local regression

The non-agent blueprint is `r1pro-classical-open-space-sim`. A reproducible test through local MCP, without a language-model call:

```bash
.classical-venv/bin/python -m dimos.robot.galaxea.r1pro.demo_classical_apartment \
  --open-space --seed 5000 --output /tmp/my-open-space-run \
  --mcp-port 10026 --zenoh-scout-addr 224.0.0.224:19492 \
  --actions pick:right:object_4 go:right:low_bench place:right:low_bench \
  --viewer --stay-open
```

Tray actions use `tray_pick::` and `tray_place::<platform>`; the arm field stays empty. A full tray sequence is `pick:right:object_4 place:right:tray tray_pick:: go::display_table tray_place::display_table`.

`--agent --say "Pick the carton with your right hand." "Put it in the tray."` runs the language agent instead and sends each sentence the way HumanCLI does, waiting for the agent to go idle and the action to finish. Pass `--model` to use a provider with working credentials. The launching shell must hold that provider's API key; the agent reports `Agent request failed` with the HTTP status otherwise.

Use a fresh output directory and unused ports for each test. The harness saves physical state and per-action outcomes. Interactive launches do not automatically execute this sequence.

GraspGenX loads its pinned checkpoint from the local HuggingFace cache without contacting the hub. If the module never logs its checkpoint paths at startup, the hub lookup is blocking; set `HF_HUB_OFFLINE=1` in the launching shell. On September 17 a half-open IPv6 route to the hub stalled startup indefinitely.
