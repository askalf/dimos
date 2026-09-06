# sim2 Hardware Emulator

The existing G1 GR00T and xArm7 planner blueprints can use `sim2` in this branch.
Their controllers remain ordinary DimOS modules. MuJoCo owns physics; robot
control uses shared memory. Cameras and lidar run independently of physics.
PimSim, task catalogs and population preparation are not involved.

## Run

```bash
uv run dimos --simulation mujoco --transport zenoh --viewer rerun --scene-package kitchen run unitree-g1-groot-wbc
uv run dimos --simulation mujoco --transport zenoh --scene-package kitchen run xarm7-planner-coordinator
```

Run one stack at a time on the default transport bus. Both open the native
MuJoCo viewer. Disable it with the module override
`--simulationmodule.viewer=false` after the blueprint name. A scene name, an
absolute XML path, or a directory containing `scene.xml` uses the same loader.
Defaults without `--scene-package` remain the small logistics/workbench scenes.

The first download of existing robot meshes and GR00T policies is separate
from measured startup. Install the existing simulation and robot dependencies.
The simulation extra requires MuJoCo 3.10 or newer for batched raycasting.

**Development Mac:** this checkout uses a locally patched Zenoh 1.9 dependency
for loopback discovery; it is not an upstream release. See
`../zenoh-python-macos-fix/LOCAL_FIX.md` in the local workspace. No router is
needed with that wheel. Existing bounded device checks can alternatively use
an explicit local router:

```bash
uv run python -m dimos.sim2.demo_smoke g1 --local-router --viewer --seconds 15 --move
uv run python -m dimos.sim2.demo_smoke xarm --local-router --viewer --seconds 15 --move
```

Add `--rerun` to include the Rerun bridge and viewer. These checks use the same
configuration parser as the CLI before deploying workers. Omit `--move` to
leave the robot holding its starting pose. The router is test setup, not a
second control path: joint commands still cross the same SHM device interface.

## Included Scenes

The eight populated scenes ship together in the existing `data/.lfs/sim2.tar.gz`
data package. Shared mesh/texture files live once in `scenes/_assets`; no
PimSim install, bundle-path environment variable, or cooking step is needed
to run them. The existing DimOS LFS mechanism obtains/extracts the archive.

| Scene name | Named entities | Movable bodies | Fixture joints | Authored robot spawns |
|---|---:|---:|---:|---|
| `kitchen` | 22 | 5 | 1 | G1, xArm7 |
| `libero-kitchen-1` | 14 | 2 | 3 | G1 |
| `libero-kitchen-9` | 15 | 3 | 1 | G1, xArm7 |
| `robocasa-kitchen-1` | 47 | 3 | 45 | G1 |
| `robocasa-kitchen-7` | 124 | 3 | 100 | G1 |
| `ithor-kitchen` | 84 | 28 | 25 | G1 |
| `procthor-house` | 90 | 51 | 15 | G1 |
| `hssd-home` | 232 | 0 | 0 | G1 |

HSSD is a furnished rigid navigation scene. Its furniture is not graspable.
The RoboCasa entries include three added movable mesh objects on an authored
counter. ProcTHOR is a multi-room house. These are scene imports, not claims
of passing the upstream benchmarks. Imported region labels are retained;
the three starter evals use explicitly authored regions in `kitchen`.

An unsupported robot spawn fails clearly; an arm is never silently placed
on the floor. Old `office` is not an alias for one of these scenes: its
legacy collision wrapper still needs a separate visual/entity conversion.

```python
from dimos.sim2.scene import list_scenes

print(list_scenes())
```

The offline maintainer script `python -m dimos.sim2.demo_prepare_scenes --source
/path/to/pimsim-starter --output data/sim2/scenes` produced this package from
the retained local source library. It is not imported by the runtime. Source
provenance is retained in each `scene.json`; source content licensing remains
subject to the original datasets' terms.

## Configure A Robot

Robot-local definitions live in:

- `dimos/robot/unitree/g1/sim2.py`: model, joint order, gains, IMU and mounts.
- `dimos/robot/manipulators/xarm/sim2.py`: native servos, gripper units and camera.

An existing blueprint selects simulated devices or real devices. It keeps its
controller, planner, perception and navigation modules:

```python
from pathlib import Path

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.manipulators.common.blueprints import coordinator, trajectory_task
from dimos.robot.manipulators.xarm.sim2 import XARM7
from dimos.sim2.blueprint import simulated_hardware, simulation_blueprint
from dimos.sim2.spec import RobotInstance

hardware = simulated_hardware(XARM7, sim_id="workbench", robot_id="arm")
devices = simulation_blueprint(
    scene=Path("/absolute/path/to/scene.xml"),
    sim_id="workbench",
    robots={"arm": RobotInstance(XARM7, xyz=(0, 0, 0.12))},
)
app = autoconnect(
    devices,
    coordinator(hardware=[hardware], tasks=[trajectory_task(hardware)]),
)
```

Adding a robot with supported controls/sensors means adding its `sim2.py`
definition and changing its existing blueprint's device selection, plus a
robot contract test and assets. There is no central robot-name switch.

Add or replace a camera on an existing configuration:

```python
from dimos.sim2.sensors.spec import Camera, Mount

XARM7_FRONT = XARM7.with_sensor(
    Camera("front", Mount("link_base", xyz=(0.1, 0, 0.3)), depth=False),
)
```

Names select sensor instances. A missing mount fails during composition.
Mount rotations use roll/pitch/yaw radians in the named body's local frame;
cameras use MuJoCo's -Z viewing direction and publish an optical-frame TF.
RGB-only and RGB-D modules have different declared ports. Repeated cameras
use `robot/sensor/port` names; multiple robots also namespace device ports.

Lidar configurations reference a concrete model such as
`dimos.sim2.sensors.lidar.models.spherical.Spherical`. New ideal ray patterns
implement the `RayPattern` contract; they need no model-name registry.

## Runtime Ownership

`SimulationModule` owns one continuously stepping `SimulationRuntime` and
disposable model snapshot. Each camera/lidar worker loads its own query model
and receives stamped integration-state frames, not a new scene per frame.
All scene geom groups are visible to camera rendering; lidar excludes only
its own robot subtree. The native viewer reads the same state snapshots.

The whole-body channel contains complete position, velocity, gains and
feed-forward torque. The adapter latches joint and IMU data together per
coordinator tick. Native xArm servos retain their original actuator model;
the gripper retains the hardware API's 0-850 units. No second PD is applied.

## Scene Interface

Use the existing `Dimos.connect()` interface (or the same module proxies in
`dimos shell`). No separate simulator client or session object is required.
All poses use metres and world coordinates; quaternion order is XYZW.
Joint angles use radians, slide-joint positions use metres.

```python
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.porcelain.dimos import Dimos
from dimos.sim2.interaction import reset_scene
from dimos.sim2.scene_types import SceneUpdate

app = Dimos.connect()
sim = app.get_module("SimulationModule")
description = sim.describe_scene()
print(description.entities.keys(), description.joints.keys(), description.regions.keys())
state = sim.scene_state()
print(state.entities["block"].pose, state.regions["tray/interior"])

sim.set_scene_state(SceneUpdate(
    poses={"block": Pose(0.30, -0.16, 0.926)},
    joints={"cabinet-1/door-hinge": 0.8},
))
reset_scene(app)  # Captured scene defaults plus robot/controller homes.
app.stop()        # Disconnect; does not stop the running blueprint.
```

Complete operator RPC surface (in addition to normal Module lifecycle):

```python
status() -> dict[str, Any]
describe_scene() -> SceneDescription
scene_state() -> SceneState
set_scene_state(update: SceneUpdate) -> SceneState
reset(initial: SceneUpdate | None = None) -> SceneState
set_spawn(robot_id: str, xyz: tuple[float, float, float],
          rpy: tuple[float, float, float] = (0, 0, 0)) -> None
set_paused(paused: bool) -> None
set_truth_enabled(enabled: bool) -> None
```

`build()` and `describe()` are internal model/snapshot bootstrap RPCs used
by sensor workers, not another scene API.

The typed records are defined in `dimos/sim2/scene_types.py`:

| Record | Fields |
|---|---|
| `SceneUpdate` | `poses: dict[entity_or_robot_id, Pose]`, `joints: dict[fixture_joint_id, float]` |
| `SceneDescription` | `format`, `id`, `entities`, `joints`, `regions`, `initial`, `spawns`, `hidden_geom_groups`, `provenance` |
| `SceneEntity` | `body`, `label`, `kind`, `movable` |
| `SceneJoint` | `joint`, `entity`, `closed`, `opened` |
| `SceneRegion` | `body`, `kind` (support/containment/navigation), local `pose`, full `size` |
| `SceneState` | `world_id`, `scene_id`, `generation`, `tick`, `sim_time`, wall `ts`, `entities`, `robots`, `joints`, `regions`, `contacts` |
| `EntityState` | world `pose`, linear `velocity`, `angular_velocity`, world `bounds_min`, `bounds_max` |
| `RegionState` | world `pose`, full `size` |

`SceneState.robots` maps instance IDs to poses; contacts are pairs of entity
IDs or robot body names. Stable scene IDs are not MuJoCo array indices.
Robot joints remain on the ordinary control interface.

### Reset And Edit Rules

`set_scene_state` validates the whole update before mutation. It changes only
existing free/mocap bodies and declared scalar fixture joints, zeroes affected
velocities, advances the command generation and publishes immediately. Model,
viewer and sensor workers remain resident. Adding assets or changing structural
scene geometry requires a new run.

`sim.reset()` restores the captured authored physics baseline, then applies
optional overrides. Overrides do not redefine the baseline. It does not cancel
application goals. Use this application-side helper for a running stack:

```python
reset_scene(
    app: Dimos,
    initial: SceneUpdate | None = None,
    *, simulation: str = "SimulationModule",
    coordinators: Sequence[str] = ("ControlCoordinator",),
    before_reset: Sequence[Callable[[Dimos], None]] = (),
    after_reset: Sequence[Callable[[Dimos], None]] = (),
) -> SceneState
```

The helper waits for controller startup, pauses physics, cancels trajectories,
deactivates controllers, runs explicit cancellation hooks, resets physics and
controller histories, runs explicit post-reset hooks, then reactivates. Failure
leaves physics paused. The starter cases supply navigation/manipulation goal
cancellation. Mapping/perception histories require hooks from their actual
owners; they are not automatically inferred or cleared. Reset affects every
robot in the world. Moving an arm's physical base does not reconfigure its
planner, so retain its authored spawn for manipulation.

## Streams And Actions

| Module | Inputs | Outputs |
|---|---|---|
| Whole-body connection | `motor_command: MotorCommandArray` | `motor_states: JointState`, `imu: Imu`, `odom: PoseStamped`, `tf: TFMessage` |
| Manipulator connection | `joint_command: JointState` | `joint_states: JointState` |
| RGB camera | none | `color_image: Image`, `camera_info: CameraInfo`, `tf: TFMessage` |
| RGB-D camera | none | RGB ports plus `depth_image: Image`, `depth_camera_info: CameraInfo` |
| Lidar | none | `pointcloud: PointCloud2` |
| SimulationModule | none | optional `sim_truth: SceneState` at 10 Hz |

Motor command input streams are consumed only in explicit
`command_source="stream"` mode. The shipped GR00T/xArm coordinators use the
direct SHM hardware adapters; sensor streams and RPCs use ordinary transport.
Robot actions remain existing navigation/manipulation RPCs such as `set_goal`,
`plan_to_poses`, `execute`, and `set_gripper_position`.

## InteractiveEval

Three authored cases live in `dimos/evals/suites/sim2_starter.py`. They perform
normal application actions. Setup uses known object coordinates; these cases
do not test RGB-D object detection.

```bash
uv run dimos evals run dimos.evals.suites.sim2_starter --tags navigation
uv run dimos evals run dimos.evals.suites.sim2_starter --tags lift
uv run dimos evals run dimos.evals.suites.sim2_starter --tags place
```

The existing `InteractiveEval` interface is preserved, with one optional field:
`action: Callable[[Dimos], None] | None`. A scripted action does not require
MCP or an LLM. Skill/instruction cases keep their existing behavior. MuJoCo
setup receives `Dimos`; DimSim setup still receives `DimSimClient`.

```python
InteractiveEval(
    id="my_case", inputs="The task instruction",
    simulator="mujoco", scene="kitchen",
    blueprint="xarm7-planner-coordinator",
    setup=prepare_scene,  # Callable[[Dimos], None]
    action=run_feature,   # Callable[[Dimos], None]; or use the existing skill field
    score=score_recording,  # Callable[[Store], float]
    interval_s=1.0, timeout_s=30,
)
```

The runner starts the existing robot blueprint with `sim2-eval-recording`,
waits for startup, calls setup, enables privileged truth recording, invokes
the action, samples the scorer and shuts down what it launched. Each case
gets its own SQLite recording and `*.setup.json` with actual initial state.
`--attach` requires that recording module in the running stack and a configured
live DB path. The scoring timeout begins after the synchronous action returns;
actions must bound their own RPC execution times.

Scorers use `fresh_states(store, initial, dwell_s=0.5)`, `inside_region`,
`contained` and `touching`. They check actual pose/contact/velocity evidence:
G1 inside its goal, block lifted with both fingers in contact, or block
released and settled wholly inside the tray. Stale or wrong-generation truth
is an error. No simulator action moves an object to make a scorer pass.

Truth is disabled by default and is not exposed as an agent skill or wired
into perception. These examples do not introduce a task catalog, distribution
service, upstream benchmark translator or task GUI.

## Scope Of This Checkpoint

Verified: G1 balance/walking, xArm joint control and gripper mapping, native
RGB-D, ideal instantaneous lidar, reset-frame invalidation, fixed-base
relocation, and independent two-robot channels/mounts.

The initial whole-body family requires one coherent control IMU. Standalone
IMU modules, timed MID360/Point-LIO input, splat rendering, automatic planner
scene obstacles, arbitrary live scene switching, full task generation/DR, and
the ten-minute latency/30-Hz-camera acceptance benchmark remain outside this
checkpoint. Authored eval availability is separate from application success.
Other robot blueprints remain on their existing backends until migrated.

### Measured Checkpoint: 2026-09-06

- 40 focused tests passed, including existing G1/xArm controls, RGB-D/lidar,
  scene edits and reset, recording, eval dispatch, and launcher shutdown.
- Mypy passed on 11 changed production files; focused Ruff checks passed.
- Eight G1 scene loads and a kitchen/xArm load rendered native RGB-D. Each
  had 32 resets preserving the compiled model. Local model construction took
  0.04-5.2 s; reset medians were 0.9-12.7 ms. These are not full blueprint
  startup timings or a sustained real-time performance benchmark.
- The scene archive is 215,281,675 bytes, expanding to 600,253,997 bytes.
  Its 7,939 mesh/texture references all resolve inside the archive.
- G1 navigation passed with recorded physical scoring. xArm lift and place
  ran through normal planner/gripper RPCs and scored zero: the grasp slipped.
  They are runnable failure examples, not solved manipulation benchmarks.
- A consecutive three-case run timed out during the third startup. That
  intermittent startup issue remains open; standalone place subsequently
  reached scoring. The existing Python 3.12 resource-tracker shutdown warning
  also remains. Do not treat this as full sequential-suite acceptance.

Local evidence: `outputs/sim2-scene-acceptance/*-camera.png`,
`~/.local/state/dimos/evals/run-20260906-163414` (G1/lift and startup failure),
and `run-20260906-164143` (standalone place). Run artifacts include the SQLite
physical recordings and exact setup snapshots. No application score was
replaced with goal acceptance or simulator-driven success.
