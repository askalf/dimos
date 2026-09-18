// Copyright 2026 Dimensional Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Onboard trajectory controllers for the RK3588: the motion2 pursuit laws
//! with no python in the tick.
//!
//! Layout: `geom` and `stamps` are shared facilities; `laws/` holds one
//! module per law: `hinted`, which the follower runs, and `seed`, the
//! permanent A/B baseline. A research generation lands by replacing
//! `laws/hinted.rs`, never by editing `laws/seed.rs` or changing the meaning
//! of an existing `geom` function.
//!
//! Rules: every law is a port, not a redesign. Its `control/laws/*.py` twin
//! is the specification and `control/test_rust_parity.py` holds the two to
//! 1e-9 per component; algorithm changes land on the python side first.
//! Single-threaded and deterministic; dependencies stay at pyo3/numpy.
//!
//! State: neither law keeps any. One that does would be a `#[pyclass]` with
//! a `reset()`, and its parity replayed as a sequence. No wall clock, no
//! unseeded randomness; the tick time arrives as an argument.
//!
//! Numerics: parity is per-operation, not per-formula: `geom.rs` keeps the
//! python's operation order and its exact tie-breaks (`argmin` takes the
//! first minimum, `searchsorted` is side='left'), and angle wrapping is IEEE
//! remainder like `math.remainder`, never `%` or `rem_euclid`.

pub mod emb;
pub mod laws;

// The path dialect and its geometry are the planner's: it encodes what these laws decode.
pub use dimos_local_planner::{clearance, geom, stamps};

#[cfg(feature = "module")]
pub mod module;

#[cfg(feature = "python")]
mod python;
