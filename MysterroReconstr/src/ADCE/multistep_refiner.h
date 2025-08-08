/* -*-c++-*- AD-Census - Copyright (C) 2020.
* Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
* https://github.com/ethan-li-coding/AD-Census
* Describe	: header of class MultiStepRefiner (Complete Fixed Version)
*/

#ifndef ADCENSUS_MULTISTEP_REFINER_H_
#define ADCENSUS_MULTISTEP_REFINER_H_

#include "adcensus_types.h"
#include <vector>
#include <utility>

using std::vector;
using std::pair;

/**
 * \brief MultiStepRefiner类，用于多步视差优化
 */
class MultiStepRefiner
{
public:
	MultiStepRefiner();
	~MultiStepRefiner();

	/**
	 * \brief 初始化
	 * \param width		影像宽度
	 * \param height	影像高度
	 * \return true:成功 false:失败
	 */
	bool Initialize(const sint32& width, const sint32& height);

	/**
	 * \brief 设置数据（标准版本）
	 * \param img_left		左影像数据，三通道
	 * \param cost			聚合代价数据
	 * \param cross_arms	交叉臂数据
	 * \param disp_left		左视差图
	 * \param disp_right	右视差图
	 */
	void SetData(const uint8* img_left, float32* cost, const CrossArm* cross_arms, 
	             float32* disp_left, float32* disp_right);
	
	/**
	 * \brief 设置数据（ADCE版本）
	 * \param disp_left		左视差图
	 * \param disp_right	右视差图
	 */
	void SetData(const float32* disp_left, const float32* disp_right);

	/**
	 * \brief 设置参数
	 * \param min_disparity				最小视差
	 * \param max_disparity				最大视差
	 * \param irv_ts					区域投票纹理阈值
	 * \param irv_th					区域投票视差阈值
	 * \param lrcheck_thres				左右一致性检验阈值
	 * \param do_lr_check				是否执行左右一致性检验
	 * \param do_region_voting			是否执行区域投票
	 * \param do_interpolating			是否执行插值
	 * \param do_discontinuity_adjustment	是否执行非连续调整
	 */
	void SetParam(const sint32& min_disparity, const sint32& max_disparity, 
	              const sint32& irv_ts, const float32& irv_th, const float32& lrcheck_thres,
				  const bool& do_lr_check, const bool& do_region_voting, 
				  const bool& do_interpolating, const bool& do_discontinuity_adjustment);

	/**
	 * \brief 执行优化
	 */
	void Refine();

	/**
	 * \brief 获取左视差图
	 * \return 左视差图指针
	 */
	const float32* GetDisparityLeft() const;

	/**
	 * \brief 获取右视差图
	 * \return 右视差图指针
	 */
	const float32* GetDisparityRight() const;

private:
	/**
	 * \brief 异常值检测
	 */
	void OutlierDetection();

	/**
	 * \brief 迭代区域投票
	 */
	void IterativeRegionVoting();

	/**
	 * \brief 适当插值
	 */
	void ProperInterpolation();
	
	/**
	 * \brief 简化插值（ADCE模式）
	 */
	void SimpleInterpolation();

	/**
	 * \brief 深度不连续调整
	 */
	void DepthDiscontinuityAdjustment();

	/**
	 * \brief 边缘检测
	 * \param edge_mask		边缘标记数组
	 * \param disp_ptr		视差图指针
	 * \param width			影像宽度
	 * \param height		影像高度
	 * \param threshold		边缘阈值
	 */
	void EdgeDetect(uint8* edge_mask, const float32* disp_ptr, 
	               const sint32& width, const sint32& height, const float32 threshold);

	/**
	 * \brief 中值滤波
	 */
	void MedianFilter();
	
	/**
	 * \brief 简化中值滤波（ADCE模式）
	 */
	void SimpleMedianFilter();

private:
	/** \brief 影像尺寸 */
	sint32 width_, height_;

	/** \brief 左影像数据 */
	const uint8* img_left_;
	
	/** \brief 聚合代价数据 */
	float32* cost_;
	
	/** \brief 交叉臂数据 - 这是缺失的关键成员变量！ */
	const CrossArm* cross_arms_;

	/** \brief 左右视差图 */
	float32* disp_left_;
	float32* disp_right_;

	/** \brief 视差范围 */
	sint32 min_disparity_, max_disparity_;

	/** \brief 算法参数 */
	sint32 irv_ts_;			// 区域投票纹理阈值
	float32 irv_th_;		// 区域投票视差阈值
	float32 lrcheck_thres_;	// 左右一致性检验阈值

	/** \brief 处理选项 */
	bool do_lr_check_;
	bool do_region_voting_;
	bool do_interpolating_;
	bool do_discontinuity_adjustment_;
	
	/** \brief 模式标志 - 这也是缺失的成员变量！ */
	bool is_adce_mode_;		// 是否为ADCE模式

	/** \brief 异常像素列表 */
	vector<pair<sint32, sint32>> occlusions_;		// 遮挡像素
	vector<pair<sint32, sint32>> mismatches_;		// 误匹配像素

	/** \brief 边缘标记 */
	vector<uint8> vec_edge_left_;

	/** \brief 优化：预分配的缓冲区 */
	vector<sint32> temp_histogram_;						// 直方图缓冲区
	vector<pair<sint32, float32>> disp_collects_;		// 视差收集缓冲区
};

#endif // ADCENSUS_MULTISTEP_REFINER_H_