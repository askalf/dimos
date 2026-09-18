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

//! The follower's law: follow the path the gait can actually walk.
//!
//! Three mechanisms over the seed, in order of what they are worth:
//!
//! 1. `walk_command`, the gait slip inverse. The twist is a REQUEST to a
//!    learned policy that under-delivers it, and the governor's creep rung sat
//!    inside the gait's dead-stall band.
//! 2. Constant time headway: the carrot distance scales with commanded speed
//!    instead of staying fixed, so the pursuit chord shrinks where the plan is
//!    tight.
//! 3. The stamped precision profile as the governor's input: the follower
//!    holds no map of its own, so the room the planner priced arrives in the
//!    path's own timestamps (`stamps::decode_ceilings`).
//!
//! (1) and (2) are held here rather than in `geom` deliberately: `seed` is the
//! frozen baseline, and sharing a mechanism with it would move it.

use crate::geom::{
    arcs_of, body_error, carrot_lerp, clearance_governor, fan_target, progress_index, yaw_command,
    Params,
};
use crate::stamps::{ceiling_ahead, decode_ceilings};

/// `Params` plus the gait calibration this law feeds forward through: the
/// embodiment's `walk_gain`/`walk_slip`/`walk_slip_ramp`, properties of the
/// gait blob and not of the law. Re-probe them on a different gait before it
/// drives hardware.
pub struct HintedParams {
    pub base: Params,
    pub walk_gain: f64,
    pub walk_slip: f64,
    /// Intended speeds below this get a proportionally smaller share of the
    /// correction, reaching zero with it. A stop request has to remain a stop.
    pub slip_ramp: f64,
}

/// The command that asks the gait for a ground speed of `want` m/s: the
/// inverse of `ground ~= gain * cmd - slip`, so the intended ground speed
/// stays the one the governor chose (`laws/hinted.py::walk_command`).
/// Identity at `want = 0`; the full inverse once `want >= ramp`.
#[inline]
pub fn walk_command(want: f64, gain: f64, slip: f64, ramp: f64) -> f64 {
    if want <= 0.0 {
        return 0.0;
    }
    let correction = (gain * want + slip) - want;
    want + correction * (want / ramp).min(1.0)
}

/// One controller tick. `clearance` is the optional per-waypoint room
/// annotation and `ts` the optional per-waypoint timestamp, which carries the
/// same information in the dialect the planner speaks (see
/// `stamps::decode_ceilings`). Returns `(vx, vy, wz)` in the body frame.
pub fn update(
    pose: (f64, f64, f64),
    path: &[[f64; 3]],
    clearance: Option<&[f64]>,
    ts: Option<&[f64]>,
    cfg: &HintedParams,
) -> (f64, f64, f64) {
    if path.len() < 2 {
        // empty path or a single-pose veto stub: there is nothing to
        // follow: hold position (the planner is saying "stop")
        return (0.0, 0.0, 0.0);
    }
    let base = &cfg.base;
    let (px, py, pyaw) = pose;
    let arcs = arcs_of(path);
    let i = progress_index(path, &arcs, px, py, pyaw);

    // Speed governor, hoisted above target selection because the lookahead
    // distance is now derived from it (constant time headway, below).
    //
    // THE SAME GOVERNOR, OFF THE STAMPS when no clearance array arrives: the
    // room ahead has not been withheld, only re-encoded, since the planner
    // stamps the required-precision profile into the path's own timestamps on
    // every replan. Decoding it puts a speed ceiling back
    // under this law at exactly the point the clearance branch occupies, so
    // the two channels are alternatives rather than layers.
    let vmax = clearance_governor(&arcs, i, clearance, base)
        .or_else(|| {
            let ceilings = decode_ceilings(ts?, path, base)?;
            Some(ceiling_ahead(&ceilings, &arcs, i, base))
        })
        .unwrap_or(base.max_speed);

    // Constant time headway: look cfg.lookahead / cfg.max_speed seconds ahead
    // instead of the seed's fixed 0.35 m. A fixed carrot chords the plan's
    // curvature to the inside of the turn, toward the obstacle the planner
    // curved around; a time headway shortens the carrot where the plan has no
    // room. The floor keeps min(k_pos * L, vmax) saturating at vmax, so this
    // costs no speed.
    let headway = base.lookahead / base.max_speed.max(1e-6);
    let look = (vmax * headway).max(vmax / base.k_pos.abs().max(1e-6));

    let (target_xy, target_yaw) =
        fan_target(path, &arcs, i, pyaw, base).unwrap_or_else(|| carrot_lerp(path, &arcs, i, look));

    // body-frame error -> velocity
    let (bx, by) = body_error(px, py, pyaw, target_xy);
    let (mut vx, mut vy) = (base.k_pos * bx, base.k_pos * by);
    let speed = vx.hypot(vy);
    if speed > 1e-12 {
        // `want` is the intended ground speed: the pursuit gain, capped by
        // the governor. Unchanged from the seed.
        let want = speed.min(vmax);
        // ...and this is what the gait has to be asked for to deliver it.
        let cmd = walk_command(want, cfg.walk_gain, cfg.walk_slip, cfg.slip_ramp);
        vx = vx / speed * cmd;
        vy = vy / speed * cmd;
    }
    (vx, vy, yaw_command(target_yaw, pyaw, base))
}
