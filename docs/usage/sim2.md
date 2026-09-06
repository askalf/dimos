# sim2 Hardware Emulator

The existing G1 GR00T and xArm7 planner blueprints can use `sim2` in this branch.
Their controllers remain ordinary DimOS modules. MuJoCo owns physics; robot
control uses shared memory. Cameras and lidar run independently of physics.
PimSim, task catalogs and population preparation are not involved.

## Run

```bash
uv run dimos --simulation mujoco --viewer rerun run unitree-g1-groot-wbc
uv run dimos --simulation mujoco run xarm7-planner-coordinator
```

Both open the native MuJoCo viewer. Disable it with the module override
`--simulationmodule.viewer=false`. To load another native robot-free scene,
pass `--scene-package /absolute/path/to/scene.xml` before `run`. A directory
containing `scene.xml` also works. Old PimSim catalog aliases are not resolved.
The defaults are small logistics/workbench scenes in `data/sim2/scenes`.

The first download of existing robot meshes and GR00T policies is separate
from measured startup. Install the existing simulation and robot dependencies.
The simulation extra requires MuJoCo 3.10 or newer for batched raycasting.

**Current launch caveat:** the default Zenoh loopback discovery timed out on
the development Mac before any simulation module started. An explicit local
Zenoh router connection passed. No transport-core change was made. These
bounded checks launch the same existing blueprints with a private ephemeral
loopback router and shut down everything they start:

```bash
uv run python -m dimos.sim2.demo_smoke g1 --local-router --viewer --seconds 15 --move
uv run python -m dimos.sim2.demo_smoke xarm --local-router --viewer --seconds 15 --move
```

Add `--rerun` to include the Rerun bridge and viewer. These checks use the same
configuration parser as the CLI before deploying workers. Omit `--move` to
leave the robot holding its starting pose. The router is test setup, not a
second control path: joint commands still cross the same SHM device interface.

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

`SimulationModule.status()`, `reset()` and `set_spawn(robot_id, xyz, rpy)` are
RPCs. Reset invalidates pre-reset commands; moving the fixed xArm changes a
mocap root without recompilation. These are physics RPCs, not a task API.
The caller must also clear controller histories through the coordinator's
normal lifecycle, including `reset_runtime_state(reactivate=True)` for GR00T.
Atomic live-stack reset coordination is not yet an acceptance claim.

## Scope Of This Checkpoint

Verified: G1 balance/walking, xArm joint control and gripper mapping, native
RGB-D, ideal instantaneous lidar, reset-frame invalidation, fixed-base
relocation, and independent two-robot channels/mounts.

The initial whole-body family requires one coherent control IMU. Standalone
IMU modules, timed MID360/Point-LIO input, splat rendering, automatic planner
scene obstacles, arbitrary scene switching, tasks, and the ten-minute
latency/30-Hz-camera acceptance benchmark are not implemented or proven here.
Other robot blueprints remain on their existing backends until migrated.
