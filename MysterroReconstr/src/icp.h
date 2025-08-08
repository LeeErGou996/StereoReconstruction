// src/icp.h
#pragma once

#include <vector>
#include "pointcloud.h"

struct Transform {
    float R[3][3];
    float t[3];
};

class RigidICP {
public:
    static Transform align(const std::vector<PointWithNormal>& source,
                           const std::vector<PointWithNormal>& target,
                           int maxIterations = 40,
                           float tolerance = 1e-5f);
};
