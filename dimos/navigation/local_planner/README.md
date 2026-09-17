# Local motion

Plan a path around obstacles over a 5-10 m horizon, and walk it. Two modules,
each pure python with a rust native twin that is what runs on the robot:

```
navigation/
  local_planner/        LocalPlanner: global route + local map + tf -> local path
    module.py           the module: carrot, replan gate, hold, clear
    native.py           the rust twin as a NativeModule
    search/             the SE(2) search: base.py protocol, se2.py, target.py
    obstacles.py        which returns are obstacles: a z-rule the body decides
    profile.py          the path dialect: precision encoded in the stamps
    viz.py              the body drawn along the plan, coloured by precision
    rust/               the crate: search, dialect, geometry; `module` feature = the twin
  trajectory_follower/  TrajectoryFollower: local path + tf -> twist
    basic/              BasicPathFollower, the plain pursuit
    fancy/              ours: module.py, native.py
    controller.py       the TrajectoryController protocol
    laws/               hinted (runs), seed (the baseline every A/B is against)
    rust/               the crate: the laws bit-exact; `module` feature = the twin
  embodiment/           one robot's measured numbers; both modules take the same one
  tf_pose.py            the body pose off tf, with a deadman
  spec.py               GlobalPlanner, BlindLocalPlanner / MapLocalPlanner, TrajectoryFollower, as ports
```

The planner is the lower layer: it owns the body-aware world and the path
dialect, and the follower decodes what it encoded. Python imports run
follower -> planner -> embodiment, and the rust crates the same way.

The planner's world is the raycaster cloud sliced by the body's own z-band; the
search is planar and its route is priced on the follower's governor, so the two
modules optimise one clock. The rust planner carries its own behavioural
invariants (`rust/tests/invariants.rs`): routes an open world, refuses a sealed
box, never hops a thin wall, answers the same way every call, memoizes nothing
across calls.

On a robot: raycaster local map -> MLS global path (remapped to `planner_path`,
it only feeds the carrot, a point `goal_lookahead_m` of arc along it) -> local
planner -> follower -> `nav_cmd_vel`. Neither module reads odometry: the pose is
the `world -> base_link` edge on tf, looked up per tick (`tf_pose.py`), and an
edge whose stamp stopped advancing for the module's deadman counts as missing.

Commands: [tools.md](tools.md), [../trajectory_follower/tools.md](../trajectory_follower/tools.md).

## Module I/O

```python
from dimos.navigation.local_planner.module import LocalPlanner

print(LocalPlanner.io(color=False))
```

```results
 ├─ local_map: PointCloud2
 ├─ planner_path: Path
 ├─ tf: TFMessage
┌┴─────────────┐
│ LocalPlanner │
└┬─────────────┘
 ├─ path: Path
```

```python
from dimos.navigation.trajectory_follower.fancy.module import TrajectoryFollower

print(TrajectoryFollower.io(color=False))
```

```results
 ├─ path: Path
 ├─ tf: TFMessage
┌┴───────────────────┐
│ TrajectoryFollower │
└┬───────────────────┘
 ├─ nav_cmd_vel: Twist
 ├─ goal_reached: Bool
```

Rules the ports carry:

- `path` timestamps are not a schedule: only their deltas carry information
  (`dt = segment / governor_speed(clearance)`, `profile.py`), so slow
  segment = tight segment. Running slower than the encoding is always legal;
  a plain-`ts` path just loses the hint, and third-party producers interoperate.
- a single-pose `path` means "hold, no safe route".
- an empty `path` means "stop": forget the plan and halt. The planner emits
  one when the global route goes away (a cancelled goal, MLS finding none).
- the follower reads no map: it decodes the room the planner encoded in the
  path stamps. `local_map` goes to the planner only.
- the follower runs `laws/hinted.py` (its rust twin on the robot); a
  config may name another `controller="module:factory"`, e.g. the `seed`
  baseline for an A/B.
