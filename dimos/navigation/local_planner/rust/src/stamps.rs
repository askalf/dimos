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

//! The precision profile the planner stamps into the path's own timestamps.
//!
//! Shared facility, not a law: this is the wire dialect, and the python side
//! of it is `control/profile.py` (`encode_precision` / `decode_ceilings`),
//! which is the specification. Any law may read it; the hinted law is the
//! one that has to, because it is the only channel through which required
//! precision reaches a follower that gets no clearance array.

use crate::geom::Params;

/// A segment shorter than this is a rotation in place, not a move: the python
/// `profile._FAN_EPS`, which this must match. Fan segments are priced by yaw
/// span, so their dt carries no precision and the decoder skips them.
pub const FAN_EPS: f64 = 1e-6;

/// The governor curve the encoder speaks: the embodiment's (`embodiment/base.py`),
/// handed across by whoever owns the body, never a per-process tuning.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Governor {
    pub max_speed: f64,
    pub min_speed: f64,
    /// Room at which full speed is granted (m).
    pub speed_clearance: f64,
    /// The embodiment's precision floor (m).
    pub floor: f64,
    /// rad/s, prices fan segments' dt.
    pub max_yaw_rate: f64,
}

/// Clearance (m) -> speed ceiling (m/s): creep at the floor, cruise with room.
///
/// A port of `profile.governor_speed`. Infinite clearance is meaningful: no
/// obstacles means nothing can touch the body, so the waypoint gets cruise.
// `max` then `min` rather than `clamp`: f64::clamp panics on NaN.
#[allow(clippy::manual_clamp)]
pub fn governor_speed(clearance: f64, gov: &Governor) -> f64 {
    let frac = (clearance - gov.floor) / (gov.speed_clearance - gov.floor);
    gov.min_speed + (gov.max_speed - gov.min_speed) * frac.max(0.0).min(1.0)
}

/// Stamp a path with its precision profile: the timestamps, in order.
///
/// A port of `profile.encode_precision`: `ts[i] - ts[i-1] = segment length /
/// governor speed`, the governor evaluated at the tighter of the segment's two
/// endpoints; `decode_ceilings` is the exact inverse. Returns the stamps
/// rather than mutating a path, since this crate has no message types.
/// `clearance` of the wrong length gives every segment cruise (the python's
/// `if len(clearance) == n` guard). Fan segments (yaw, no displacement) are
/// priced by yaw span at `max_yaw_rate`, which is why the decoder skips them.
pub fn encode_precision(path: &[[f64; 3]], clearance: &[f64], t0: f64, gov: &Governor) -> Vec<f64> {
    let n = path.len();
    if n == 0 {
        return Vec::new();
    }
    let use_clearance = clearance.len() == n;
    let speed = |k: usize| {
        if use_clearance {
            governor_speed(clearance[k], gov)
        } else {
            gov.max_speed
        }
    };
    let mut ts = vec![t0; n];
    let mut t = t0;
    for k in 1..n {
        let (dx, dy) = (path[k][0] - path[k - 1][0], path[k][1] - path[k - 1][1]);
        let ds = (dx * dx + dy * dy).sqrt();
        if ds < FAN_EPS {
            // `np.remainder` (floor-mod), not the IEEE remainder the laws use
            // for angle_diff: mirrors the python statement exactly.
            let dyaw = path[k][2] - path[k - 1][2];
            let wrapped = (dyaw + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
                - std::f64::consts::PI;
            t += wrapped.abs() / gov.max_yaw_rate;
        } else {
            t += ds / speed(k - 1).min(speed(k));
        }
        ts[k] = t;
    }
    ts
}

/// Per-waypoint speed ceiling (m/s) recovered from the stamps, or `None` when
/// the producer does not speak the dialect.
///
/// A port of `profile.decode_ceilings`, statement for statement: `ds/dt`
/// recovers `min(gov(clr[i-1]), gov(clr[i]))` in m/s. Fewer than two poses,
/// or stamps that are flat or go backwards, return `None`; fan segments
/// inherit the previous ceiling; the result is clipped into
/// `[min_speed, max_speed]`, so a stamp can only ask for more care than cruise.
/// A per-waypoint ceiling keyed to arc position, not a schedule: absolute
/// times never enter. `ds` is the raw segment, as in the python, not a
/// difference of `arcs`.
pub fn decode_ceilings(ts: &[f64], path: &[[f64; 3]], cfg: &Params) -> Option<Vec<f64>> {
    let n = path.len();
    if n < 2 || ts.len() != n {
        return None;
    }
    // `np.any(dt < 0) or not np.any(dt > 0)`: unstamped (all-equal ts) and
    // non-monotone paths are rejected outright.
    let mut any_positive = false;
    for k in 1..n {
        let dt = ts[k] - ts[k - 1];
        if dt < 0.0 {
            return None;
        }
        if dt > 0.0 {
            any_positive = true;
        }
    }
    if !any_positive {
        return None;
    }
    // min/max rather than the raw fields: `f64::clamp` panics when the bounds
    // cross, and a config is free to set min_speed above max_speed.
    let lo = cfg.min_speed.min(cfg.max_speed);
    let hi = cfg.max_speed.max(cfg.min_speed);
    let mut out = vec![hi; n];
    let mut prev = hi;
    for k in 1..n {
        let (dx, dy) = (path[k][0] - path[k - 1][0], path[k][1] - path[k - 1][1]);
        let ds = (dx * dx + dy * dy).sqrt();
        let dt = ts[k] - ts[k - 1];
        if ds >= FAN_EPS && dt > 0.0 {
            let v = ds / dt;
            // NaN would propagate straight out through the twist; the python
            // cannot produce one here because its inputs are the encoder's,
            // but this law takes whatever the wire hands it.
            if v.is_finite() {
                prev = v.clamp(lo, hi);
            }
        }
        out[k] = prev;
    }
    out[0] = out[1];
    Some(out)
}

/// The tightest decoded ceiling within `speed_lookahead` of `arcs[i]`.
///
/// Scans from `i + 1`: a decoded ceiling belongs to the segment ending at its
/// waypoint, so `ceilings[k]` already carries `clr[k-1]` and `[i+1 ..]`
/// reproduces `gov(min clr over [i ..])` exactly.
pub fn ceiling_ahead(ceilings: &[f64], arcs: &[f64], i: usize, cfg: &Params) -> f64 {
    let n = arcs.len();
    let hi = arcs[i] + cfg.speed_lookahead;
    // The segment about to be traversed always counts, even if a degenerate
    // `speed_lookahead` would exclude it; at the end of the plan there is no
    // next segment and the last ceiling stands.
    let mut room = ceilings[(i + 1).min(n - 1)];
    for k in (i + 1)..n {
        if arcs[k] > hi {
            break; // arcs are non-decreasing, so nothing later qualifies
        }
        if ceilings[k] < room {
            room = ceilings[k];
        }
    }
    room
}
