# Memory World

First-person VR exploration of a recorded robot memory. The viewer builds a
height-coloured voxel map from the recording's lidar stream and overlays image
captures, odometry, and the places an answer names.

Questions are answered from the recording's own CLIP/SigLIP frame embeddings:
the question goes through the model's text tower, the closest frames' hot
patches are raycast through depth, and the resulting points are clustered into
places. `DEMO.md` is the walkthrough.

## Running

```bash
uv run dimos run memory-world-module --memoryworldmodule.store-path recording.mcap
```

`~/Commands/memworld <recording>` wraps this with the right paths and a single
server lock.

The `memory-world-agent` blueprint starts the same viewer with an MCP server and
agent alongside it, so the question can come from a chat rather than the bar:

```bash
uv run dimos run memory-world-agent \
  --memoryworldmodule.store-path hk_building_park.db \
  --memoryworldmodule.background-mode passthrough
```

Set the API key required by the configured model, then run `uv run dimos
humancli` in another terminal. The agent's one tool is `find_in_memory`.

Useful configuration flags include `--voxel-size`, `--max-points`,
`--n-voxel-scans`, `--n-image-markers`, and
`--background-mode {black,passthrough}`. Set `--n-voxel-scans 0` to consume
every lidar frame.

## Viewing in VR

1. On the Quest 3, open the browser and go to `https://<host-ip>:8443/memory_world`.
2. Accept the self-signed cert, tap **Connect**, then enter VR.
3. Use the left thumbstick to walk/strafe and the right thumbstick to turn.
4. Hold and release the right trigger to teleport.
5. Hold both grips, or pinch both hands, and change their separation to scale.
6. Press left X to toggle images, left Y to reset, and right B to toggle voxels.

A minimap shows the current position and heading. The same cloud-derived map is
projected onto the ground.

Host and headset must be on the same Wi-Fi / LAN.
