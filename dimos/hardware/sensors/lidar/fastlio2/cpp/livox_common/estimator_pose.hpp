// Copyright 2026 Dimensional Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <vector>

namespace dimos {

inline constexpr double kQuatNormTolerance = 1e-3;

inline bool has_estimate(const std::vector<double>& pose) {
    if (pose.size() != 7) {
        return false;
    }
    const double qn = pose[3] * pose[3] + pose[4] * pose[4] + pose[5] * pose[5] +
                      pose[6] * pose[6];
    return std::abs(qn - 1.0) < kQuatNormTolerance;
}

}  // namespace dimos
