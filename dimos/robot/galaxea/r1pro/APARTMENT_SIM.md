# Interactive apartment ACT development

The apartment integration is experimental. Navigation to the dining table has
passed a full DimOS/MCP simulation test. Existing ACT weights still fail the
apartment pick tests; the apartment preview has not been published.

The intended launch is `dimos run r1pro-apartment-sim-agent`, followed by
`dimos humancli` in a second terminal. This requires a validated local
`recordings/r1pro-act-task/policy-apartment-preview` bundle, which is not yet
available. The older object/primitive preview remains separate.

Each launch samples five small household props on physical worktable, kitchen
and dining-table patches. Seeded runs reproduce object identity, dimensions and
positions. The props are procedural bottles, hollow cups, cartons, glue sticks
and toy blocks. They do not establish arbitrary-object generalization.

Commands are independent: pick lifts and holds, place releases onto the named
support, and go_to transports held items without releasing. Auto hand selection
compares free hands; explicit left/right is preserved. Placement auto-selects a
hand only when exactly one is occupied. With two occupied hands, specify which
object or hand to place. Geometry is currently simulator ground truth.

Before ACT, the stack checks arm reach and the complete vertical manipulation
corridor, then uses DimOS positioning to enter that workspace. Training coverage
is reported separately from physical reachability. Empty placement candidates
must fit on a measured support without overlapping objects or tray geometry.
The current high kitchen poses need additional demonstrations. Navigation uses
the static environment pointcloud, KronkNav and a holonomic coordinator task;
complete swept robot and held-object geometry is checked before execution.

For development, `demo_primitive_interactive --apartment --policies PATH` runs
scripted actions through the local MCP interface without calling an external
LLM. It accepts a scene package and seed for reproducible checks. Review the
saved result; command acceptance and IK success are not ACT success.

Current collection status:

```bash
cat /home/mustafa/dimos-wt/r1pro-act-sim/recordings/r1pro-act-task/jobs/apartment-appearance-v2/collection/status.json
```

See [the handoff](../../../../openspec/changes/r1pro-act-house-sim/handoffs-random-objects.md)
for job evidence and unfinished work.
