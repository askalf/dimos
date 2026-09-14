# Microduck simulation

Run the upstream Microduck model and pretrained walking policy in CPU MuJoCo:

```bash
uv sync --extra sim --extra cpu --inexact
dimos --simulation run microduck-sim --daemon
dimos status
dimos shell
```

For keyboard control, run `dimos --simulation run microduck-sim-keyboard --daemon`
instead. Click the **Keyboard Teleop** window to focus it, then hold W/S to move
forward/backward, A/D to turn, or Q/E to strafe. Release the movement keys to stop.
Space stops all motion, Shift boosts speed, Ctrl slows down, and Escape closes the
keyboard window. The MuJoCo window shows the robot; movement keys go to the keyboard
window. When idle, keyboard teleop leaves RPC velocity commands available.

In Python, command a short walk or reset the running simulation:

```python
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.porcelain.dimos import Dimos

app = Dimos.connect()
app.MicroduckSim.move(Twist(linear=[0.3, 0, 0]), duration=3.0)
app.MicroduckSim.get_state()
app.MicroduckSim.reset()
app.stop()  # Disconnect this client; the daemon keeps running.
```

`dimos stop` closes the simulation and its MuJoCo window. For a machine without a
display, append `--microducksim.headless=true` to the run command.

The first run caches pinned revisions of the
[Microduck model](https://github.com/pollen-robotics/microduck_rl) and
[official policies](https://huggingface.co/pollen-robotics/microduck-policies).
Git LFS is required for the policy download. To use existing local assets, append
`--microducksim.scene-path=/path/to/microduck_rl/src/mjlab_microduck/robot/microduck/scene.xml`
and `--microducksim.policy-path=/path/to/alpha_walking.onnx`.

The module accepts `cmd_vel` and publishes `joint_state`, `odom`, and `tf`. Streamed
commands expire unless refreshed; the `move` RPC can specify a duration. This
blueprint uses the scene's MuJoCo position actuators and the policy's metadata for
joint ordering and reference angles. It does not emulate the upstream BAM motor
model. The walking model has 14 actuated joints and no controllable mouth.
