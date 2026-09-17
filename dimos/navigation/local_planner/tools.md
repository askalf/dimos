# Tools

Two packages: `local_planner/` plans, `trajectory_follower/` follows a plan;
each is a python module with a rust native twin. Run from the repo root. The
follower's menu is [../trajectory_follower/tools.md](../trajectory_follower/tools.md).

What is here is what runs on a robot, and the tests that hold it:
python-vs-rust parity on the laws, the crate's own behavioural invariants, and
the module wiring. The benchmarks are not on this branch.

```bash
# the module and its tests
uv run pytest dimos/navigation/local_planner -q
uv run mypy dimos/navigation/local_planner

# the crate's behavioural invariants: routes an open world, refuses a sealed
# box, never hops a thin wall, same answer every call, no cross-call memo.
cargo test --release --test invariants --manifest-path dimos/navigation/local_planner/rust/Cargo.toml

# the native twin's own tests
cargo test --features module --manifest-path dimos/navigation/local_planner/rust/Cargo.toml

# the pyo3 extension is test-only (the planner parity); the python module runs the python search
uv run maturin develop --uv --release --features python -m dimos/navigation/local_planner/rust/Cargo.toml

# the twin binary, which LocalPlannerNative builds itself on first start
cargo build --release --features module --manifest-path dimos/navigation/local_planner/rust/Cargo.toml

# the loop on a real go2 (robot runs the go2web zenoh bridge):
# raycaster -> MLS carrot -> local planner -> follower -> cmd_vel
dimos run go2-zenoh-motion
```
