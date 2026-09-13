# Native PICO full-body teleoperation

This connector uses the public **XRoboToolkit PICO APK and PC Service** to feed
the existing G1 SONIC teleop task. It does not request WebXR body tracking or use
an enterprise account. Native body tracking and controller input have been tested
with a consumer PICO 4 Ultra and calibrated Motion Trackers.

```text
PICO 4 Ultra + calibrated Motion Trackers
  → XRoboToolkit APK → XRoboToolkit PC Service
  → PicoTeleopModule → BodyTrackingSnapshot / Buttons / Twist
  → existing G1 SONIC retargeter and controller → MuJoCo G1
```

The existing retargeter, pose buffer, A+X mode toggle, policy models and planner
fallback are reused. This connector currently supplies full-body tracking and
controller buttons/sticks. In simulation, `PicoVideoModule` also sends the head
camera to the stock APK's **Remote Vision** receiver. No custom APK or browser
session is needed. Separate hand/controller pose streams for arm IK are outside
this connector's scope.

## PC setup

Install the DimOS `xrobotoolkit` extra alongside your existing SONIC/simulation
extras. For an environment already running SONIC, this preserves its packages:

```bash
uv sync --inexact --extra xrobotoolkit
```

Then run the blueprint below. On **Ubuntu 22.04 or 24.04 x86_64**, its
`PicoTeleopModule.build()` prepares the matching public
[PC Service package](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/tag/v1.0.0)
before any modules start. It reuses existing files or downloads the pinned
package, checks its SHA-256, and extracts it into
`~/.cache/dimos/xrobotoolkit-pc-service/` (respecting `$XDG_CACHE_HOME`). Versions
and checksums live in [`install.py`](install.py). Extraction uses `dpkg-deb`,
provided by Ubuntu's `dpkg` package, without sudo or running package installation
scripts. Subsequent runs reuse the cache without network access. Concurrent
builds share a lock; failed downloads/extractions never become the active cache.

The module then starts the PC Service, waits for its gRPC heartbeat, and
stops/reaps its process during teardown, including failed startup. No separate
download or service-launch command is needed for a normal blueprint run.
Build/setup failure prevents the stack from starting;
vendor output goes to `~/.local/state/dimos/pico/pc-service-<worker-pid>.log`
(or `$XDG_STATE_HOME/dimos/pico`).

Existing system files at `/opt/apps/roboticsservice` and earlier home-directory
extractions under `$XDG_DATA_HOME/dimos/xrobotoolkit-pc-service[-1.0.0]` are
reused. For a custom or offline extraction, pass
`--pc-service-dir /path/to/opt/apps/roboticsservice` after the blueprint name.
This explicit path must already contain an executable `RoboticsServiceProcess`;
DimOS does not download into or replace it. Other platforms require compatible
local files or an external service.

DimOS connects locally on `127.0.0.1:60061`; this is **not** the address entered
on the headset. If that service is already responding, the module uses it and
leaves shutdown to its existing owner. `--manage-pc-service false` explicitly uses
an external service and skips local setup. Remote hosts and non-default gRPC
ports also skip setup and use external services. `tracking_status()` reports the
owned `pc_service_pid` and its exit
code. If an owned service crashes, input is invalidated through the existing
connection/freshness checks; restart the blueprint to start a new service.

The C++/Python SDK binding is unnecessary: DimOS uses the same gRPC feedback
subscription as the official C SDK. It does not compile or load the native SDK.

## Headset setup

1. Install and open the public [XRoboToolkit PICO v1.1.1 APK](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/tag/v1.1.1).
2. Keep the headset and PC on the same LAN. Connect the APK to the **PC's LAN
   address**, available from `hostname -I` on the PC. Do not enter a browser URL.
3. Pair, wear and calibrate the Motion Trackers using PICO's body-tracking
   calibration. Select **Body** tracking in XRoboToolkit, enable **Controller**
   data, and turn **Send** on. Keep the APK in the foreground.
4. Turn off the APK's **Switch w/A Button** option: A is part of the SONIC A+X
   toggle, so it must not also turn packet sending off.
5. Release A and X. The connector waits for complete, advancing body samples
   and the released buttons before enabling operator commands.

## Check tracking, then run simulation

First check the input without starting a robot:

```bash
uv run --no-sync dimos --transport zenoh --viewer none run teleop-pico-body-tracking
```

The `PICO tracking` log reports service connection, device ID, body availability,
sample counts, sample age, timestamps and the reason for waiting. Look for
`pc_service_connected=true`, `state=tracking`, and an increasing `body_frames`
count. `PicoTeleopModule.tracking_status()` exposes the same data over module RPC.
Stop this diagnostic run with Ctrl-C before starting the G1 blueprint.

```bash
uv run --no-sync dimos --transport zenoh --simulation mujoco --viewer rerun \
  run unitree-g1-sonic-pico-teleop
```

Use `--viewer none` to skip Rerun. The existing
`--sonic-pipeline sonic-low-latency` option is also supported after the blueprint
name. For a different PC-service endpoint, set the global
`--xrobotoolkit-host` / `--xrobotoolkit-port` flags before `run`.
To select a particular headset, pass `--device-id <service-device-id>` after the
blueprint name. Otherwise the first headset is selected and retained until the
module restarts; another headset cannot take it over.

In simulation, let the pose buffer fill, then press **A+X together** to enter
POSE. Release both and press A+X again to return to PLANNER. In PLANNER, left
stick Y commands forward/back motion and right stick X turns. In POSE, the
body controls motion and the right stick adjusts heading. Holding the right
stick click sends zero velocity. Check these directions with small movements
during the first live test.

**A+B+X+Y together latches a damping stop.** Releasing the buttons, pressing A+X,
or receiving fresh tracking cannot resume motion. The stop buttons bypass the
connector's body/focus/reconnect gates. On hardware they also reach the final
DDS publisher directly, independently of policy inference.

The computer equivalent is:

```bash
uv run --no-sync dimos --transport zenoh hardware g1 estop
```

The stop sets all 29 motors to zero stiffness/feedforward and zero desired
velocity, with damping gain `kd=8`. It also trips on measured overspeed in either
direction (default 35 rad/s), policy errors/non-finite values, and real-hardware
feedback or command loss (default 0.1 seconds). Background planner inference has
a one-second deadline. Deployment defaults live in
[`sonic_safety.py`](../../control/tasks/g1_sonic_wbc_task/sonic_safety.py);
they have offline test coverage, not real-G1 validation.

Inspect the reason with `dimos hardware g1 status` or the connection's
`command_stream_status()` RPC. After addressing it and supporting the robot,
deliberately restart the whole stack to recover. Hardware then starts unarmed
with learned-policy output in dry-run and requires `arm` and `enable` again.
An ordinary `disable` still means joint hold; it does not clear a damping stop.
Damping removes active balance and is not power isolation. Keep the Unitree
physical stop available during hardware tests.

Hardware dry-run still acquires joint hold and executes the arming ramp. During
preview, the fixed hold target is refreshed while learned targets are suppressed,
so a stalled policy can be distinguished from healthy dry-run operation.

Both the MuJoCo window and Rerun show the live robot on the PC. Rerun animates
the existing G1 URDF from measured joint feedback and the simulator's root pose,
even in PLANNER or while waiting for headset tracking. The active SONIC human
reference appears during POSE/POSE_TRANSITION. The bridge keeps only the newest
pending updates and bounds recording memory to 256 MB. `--viewer none` disables
Rerun while retaining the MuJoCo window.

## View the simulated camera in the stock APK

Keep XRoboToolkit in the foreground with tracking sending enabled. Under
**Remote Vision**, select **PICO4U**, tap **Listen**, and enter the **PC's LAN IP**
(no URL or port). The built-in profile works without changing headset files.
Press **B** to switch between side-by-side and per-eye display modes.

The sender duplicates the simulated mono camera into both eyes, preserving its
aspect ratio. This is a camera view, not stereo depth. The stock **ZEDMINI**
profile is also accepted, with letterboxing to preserve the source aspect ratio.
Neither profile accesses the PICO's own passthrough cameras.

The simulation blueprint starts a request listener on `xrobotoolkit_video_port`
(default TCP 13579) and sends H.264 to the requesting headset's video port
(the stock APK uses TCP 12345). `--listen-host` can restrict the listener to a
specific PC interface. Encoding uses PyAV, included in the `xrobotoolkit` extra;
no additional vendor video service, Jetson or local APK build is required.
`--max-fps` and `--max-bitrate` following the blueprint name cap video load.

`PicoVideoModule.video_status()` reports the listener, connected peer, frames and
bytes sent, and errors. A growing frame count proves delivery to the headset's
TCP connection; headset display still needs visual confirmation. Toggle Listen
off/on to reconnect. Closing video leaves body/controller tracking independent.
The hardware blueprint does not add this video module automatically: a real
camera must be wired separately.

Do not open an immersive browser alongside the native app: the tested consumer
headset asks to close one. This blueprint uses the native app for all headset
input and viewing; the existing WebXR implementation remains a separate option
on compatible devices.

## Freshness and recovery

- Every frame must contain all 24 finite joint poses. The PICO 4 Ultra stream
  observed with the public APK populates the IMU timestamp `t` on the head joint
  and leaves the other 23 timestamps zero, even as all body poses update.
  At least one body joint clock must be populated; all populated clocks must
  advance before another frame is published. `body_clock_joints` reports which
  clocks are being used. A change in that set restarts tracking acquisition.
- Duplicate packets and cached body poses cannot refresh tracking. Tracking
  loss expires within one second by default; missing Body, loss of app focus,
  malformed packets or a disconnected service explicitly clear the body and
  send zero buttons/velocity. Commands are gated on usable body tracking.
- Clock resets and reconnects establish a new reference generation. The existing
  SONIC task falls back to PLANNER. Release A and X, let tracking refill, then
  press them again to engage. Recentring while sending is not signalled by the
  released APK; stop the DimOS run before recentering and restart afterwards.
- Joint `t` values are treated as opaque sensor timestamps. Packet
  `timeStampNs` supplies frame intervals, aligned to the PC on the first packet.
  Reported age is relative to that alignment, **not measured headset-to-PC
  latency**. Accumulating packet delay is rejected; initial network delay cannot
  be measured from this protocol alone.

`waiting_for_body_clock` means the first complete body sample arrived but a
second sample with advancing joint clocks has not. `body_stale` means they
stopped advancing. `invalid_packet` includes a `last_error` diagnostic. If this
headset firmware returns no usable body joint timestamps or invalid-but-advancing
poses, the released APK needs a public source change exposing reliable validity
metadata; the connector deliberately does not invent freshness. That remains a
live acceptance check, along with calibration, alignment, turning and recovery.

## Protocol maintenance and checks

The wire-compatible subset in `proto/tracking.proto` preserves method names and
field numbers from the [released Linux service schema](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/blob/6386009475369615fa984bca0a6e9902cd81eb2a/RoboticsService/PXREAService/linux_x86/PXREAService.proto).
JSON fields and native pose conventions follow the [v1.1.1 tracking source](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/blob/9f775b535d781618bd2bb7ef8d6c414c0531387c/Assets/Scripts/TrackingData.cs).
Video requests follow the released [camera serializer](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/blob/v1.1.1/Assets/Scripts/Network/CameraRequestSerializer.cs),
and H.264 framing follows the official [TCP sender](https://github.com/XR-Robotics/XRoboToolkit-Orin-Video-Sender/blob/main/main_web_gst.cpp).
Regenerate checked-in protobuf files from the repository root:

```bash
uvx --from grpcio-tools==1.78.0 python -m grpc_tools.protoc -I . \
  --python_out=. --pyi_out=. dimos/teleop/pico/proto/tracking.proto
uv run --no-sync pytest dimos/teleop/pico
```

Tests cover a real loopback gRPC server, packet validation, skeleton conversion
into the existing NVIDIA retargeter, clocks, loss, reconnect and module shutdown.
Video tests exercise fragmented requests over real loopback TCP, H.264 decoding,
color and eye layout, reconnects, and closing active connections on shutdown.
They establish readiness for a live PICO → SONIC → MuJoCo test; they do not
establish hardware tracking quality.
