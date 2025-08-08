// src/utils/kdtree.h
#pragma once

#include "../pointcloud.h"
#include "nanoflann.hpp"
#include <vector>
#include <cstddef>

using namespace nanoflann;

// 点云适配器
struct PointCloudAdaptor {
    const std::vector<PointWithNormal>& pts;

    PointCloudAdaptor(const std::vector<PointWithNormal>& points) : pts(points) {}

    inline size_t kdtree_get_point_count() const { return pts.size(); }

    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        if (dim == 0) return pts[idx].point.x;
        if (dim == 1) return pts[idx].point.y;
        return pts[idx].point.z;
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

// 包装类
class KDTreeWrapper {
public:
    KDTreeWrapper(const std::vector<PointWithNormal>& points)
        : adaptor(points), index(3, adaptor, KDTreeSingleIndexAdaptorParams(10)) {
        index.buildIndex();
        ref = &points;
    }

    int findClosest(const Point3D& query) const {
        float query_pt[3] = { query.x, query.y, query.z };
        size_t ret_index;
        float out_dist_sqr;
        KNNResultSet<float> resultSet(1);
        resultSet.init(&ret_index, &out_dist_sqr);
        index.findNeighbors(resultSet, query_pt, nanoflann::SearchParameters(10));
        return static_cast<int>(ret_index);
    }

    void kNearest(const Point3D& query, int k, std::vector<int>& indices) const {
        float query_pt[3] = { query.x, query.y, query.z };
        std::vector<size_t> ret_indexes(k);
        std::vector<float> out_dists_sqr(k);

        KNNResultSet<float> resultSet(k);
        resultSet.init(ret_indexes.data(), out_dists_sqr.data());
        index.findNeighbors(resultSet, query_pt, nanoflann::SearchParameters(10));

        indices.clear();
        for (int i = 0; i < k; ++i)
            indices.push_back(static_cast<int>(ret_indexes[i]));
    }

private:
    const std::vector<PointWithNormal>* ref;
    PointCloudAdaptor adaptor;
    typedef KDTreeSingleIndexAdaptor<
        L2_Simple_Adaptor<float, PointCloudAdaptor>,
        PointCloudAdaptor,
        3
    > KDTree;
    KDTree index;
};
