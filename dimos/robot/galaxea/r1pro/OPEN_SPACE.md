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

This scene uses the same privileged simulation perception and contact checks as the [classical apartment](CLASSICAL_APARTMENT.md). It simplifies navigation geometry; it does not establish arbitrary-object or hardware reliability.

## Verified run

Seed 5000 completed ten consecutive commands through local MCP: right-hand carton pick, low-bench delivery and re-pick, high-counter delivery, navigation to the display table, left-hand cup pick, and delivery to the tall table. Both final objects were physically supported, released, upright and settled. A separate full GLFW viewer startup passed at seed 5001. These checks did not call the language model.

Evidence is saved locally in `recordings/r1pro-classical-open-space/seed-5000/validation.json`, with per-action states and a scene preview in the same directory. The focused suite passed 68 tests; strict type checking passed on ten changed production sources.

## Local regression

The non-agent blueprint is `r1pro-classical-open-space-sim`. A reproducible test through local MCP, without a language-model call:

```bash
.classical-venv/bin/python -m dimos.robot.galaxea.r1pro.demo_classical_apartment \
  --open-space --seed 5000 --output /tmp/my-open-space-run \
  --mcp-port 10026 --zenoh-scout-addr 224.0.0.224:19492 \
  --actions pick:right:object_4 go:right:low_bench place:right:low_bench \
  --viewer --stay-open
```

Use a fresh output directory and unused ports for each test. The harness saves physical state and per-action outcomes. Interactive launches do not automatically execute this sequence.
