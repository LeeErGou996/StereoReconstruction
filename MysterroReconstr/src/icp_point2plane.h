// src/icp_point2plane.h
#pragma once
#include "icp.h"
#include "pointcloud.h"
#include <vector>

class PointToPlaneICP {
public:
    static Transform align(const std::vector<PointWithNormal>& source,
                           const std::vector<PointWithNormal>& target,
                           int maxIterations = 20,
                           float maxMatchDist = 0.05f,
                           float stopThreshold = 1e-4);
};
