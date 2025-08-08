/* -*-c++-*- SemiGlobalMatching - Copyright (C) 2020.
* Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
* https://github.com/ethan-li-coding/SemiGlobalMatching
* Describe	: header of semi-global matching class (Linux version)
*/

#ifndef SEMI_GLOBAL_MATCHING_H
#define SEMI_GLOBAL_MATCHING_H

#include <cstdint>
#include <vector>
#include <utility>
#include "sgm_types.h"

// 安全删除宏
#define SAFE_DELETE(p) { if(p) { delete[] (p); (p) = nullptr; } }

/**
 * \brief census窗口类型
 */
enum CensusSize {
    Census5x5 = 0,
    Census9x7 = 1
};

/**
 * \brief SGM参数结构体
 */
struct SGMOption {
    uint8	num_paths;			// 聚合路径数： 4 or 8
    sint32	min_disparity;		// 最小视差
    sint32	max_disparity;		// 最大视差

    CensusSize	census_size;	// census窗口尺寸

    bool	is_check_unique;	// 是否检查唯一性
    float32	uniqueness_ratio;	// 唯一性约束阈值 （最小代价-次最小代价)/最小代价 > 阈值 为有效估计

    bool	is_check_lr;		// 是否检查左右一致性
    float32	lrcheck_thres;		// 左右一致性约束阈值

    bool	is_remove_speckles;	// 是否移除小的连通区
    sint32	min_speckle_aera;	// 最小连通区面积（单位：像素）

    bool	is_fill_holes;		// 是否填充视差空洞

    sint32	p1;				// 惩罚项P1
    sint32	p2_init;		// 惩罚项P2

    SGMOption(): num_paths(8), min_disparity(0), max_disparity(64), census_size(Census5x5),
                 is_check_unique(true), uniqueness_ratio(0.95f),
                 is_check_lr(true), lrcheck_thres(1.0f),
                 is_remove_speckles(true), min_speckle_aera(20),
                 is_fill_holes(true),
                 p1(10), p2_init(150) { }
};

/**
 * \brief Semi-Global Matching 类
 */
class SemiGlobalMatching
{
public:
    SemiGlobalMatching();
    ~SemiGlobalMatching();

public:
    /**
     * \brief 初始化
     * \param width			影像宽度
     * \param height		影像高度
     * \param option		SGM参数
     * \return true:成功
     */
    bool Initialize(const sint32& width, const sint32& height, const SGMOption& option);

    /**
     * \brief 执行匹配
     * \param img_left		左影像数据指针，单通道
     * \param img_right		右影像数据指针，单通道
     * \param disp_left		左影像视差图，float32
     * \return true:成功
     */
    bool Match(const uint8* img_left, const uint8* img_right, float32* disp_left);

    /**
     * \brief 重设
     * \param width			影像宽度
     * \param height		影像高度
     * \param option		SGM参数
     * \return true:成功
     */
    bool Reset(const uint32& width, const uint32& height, const SGMOption& option);

private:
    /**
     * \brief 释放内存
     */
    void Release();

    /**
     * \brief Census变换
     */
    void CensusTransform() const;

    /**
     * \brief 代价计算
     */
    void ComputeCost() const;

    /**
     * \brief 代价聚合
     */
    void CostAggregation() const;

    /**
     * \brief 视差计算
     */
    void ComputeDisparity() const;

    /**
     * \brief 右影像视差计算
     */
    void ComputeDisparityRight() const;

    /**
     * \brief 一致性检查
     */
    void LRCheck();

    /**
     * \brief 视差图洞填充
     */
    void FillHolesInDispMap();

private:
    /** \brief SGM参数 */
    SGMOption option_;

    /** \brief 影像宽度 */
    sint32 width_;

    /** \brief 影像高度 */
    sint32 height_;

    /** \brief 左影像数据 */
    const uint8* img_left_;

    /** \brief 右影像数据 */
    const uint8* img_right_;

    /** \brief 左影像census值 */
    void* census_left_;

    /** \brief 右影像census值 */
    void* census_right_;

    /** \brief 初始匹配代价 */
    uint8* cost_init_;

    /** \brief 聚合匹配代价 */
    uint16* cost_aggr_;

    /** \brief 聚合匹配代价-方向1 */
    uint8* cost_aggr_1_;

    /** \brief 聚合匹配代价-方向2 */
    uint8* cost_aggr_2_;

    /** \brief 聚合匹配代价-方向3 */
    uint8* cost_aggr_3_;

    /** \brief 聚合匹配代价-方向4 */
    uint8* cost_aggr_4_;

    /** \brief 聚合匹配代价-方向5 */
    uint8* cost_aggr_5_;

    /** \brief 聚合匹配代价-方向6 */
    uint8* cost_aggr_6_;

    /** \brief 聚合匹配代价-方向7 */
    uint8* cost_aggr_7_;

    /** \brief 聚合匹配代价-方向8 */
    uint8* cost_aggr_8_;

    /** \brief 左影像视差图 */
    float32* disp_left_;

    /** \brief 右影像视差图 */
    float32* disp_right_;

    /** \brief 遮挡区像素集合（行列号集合） */
    std::vector<std::pair<sint32, sint32>> occlusions_;

    /** \brief 误匹配区像素集合（行列号集合） */
    std::vector<std::pair<sint32, sint32>> mismatches_;

    /** \brief 是否成功初始化标志 */
    bool is_initialized_;
};

#endif // SEMI_GLOBAL_MATCHING_H