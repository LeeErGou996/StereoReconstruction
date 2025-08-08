/* -*-c++-*- AD-Census - Copyright (C) 2020.
* Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
* https://github.com/ethan-li-coding/AD-Census
* Describe	: header of class ScanlineOptimizer (Optimized Version)
*/

#ifndef ADCENSUS_SCANLINE_OPTIMIZER_H_
#define ADCENSUS_SCANLINE_OPTIMIZER_H_

#include "adcensus_types.h"
#include <vector>

/**
 * \brief ScanlineOptimizer类，用于扫描线优化
 */
class ScanlineOptimizer
{
public:
	ScanlineOptimizer();
	~ScanlineOptimizer();

	/**
	 * \brief 设置数据（标准版本）
	 * \param img_left		左影像，三通道
	 * \param img_right		右影像，三通道  
	 * \param cost_init		初始代价数据
	 * \param cost_aggr		聚合代价数据
	 */
	void SetData(const uint8* img_left, const uint8* img_right, float32* cost_init, float32* cost_aggr);
	
	/**
	 * \brief 设置数据（ADCE版本）
	 * \param cost_left		左视图代价数据
	 * \param cost_right	右视图代价数据
	 */
	void SetData(const float32* cost_left, const float32* cost_right);

	/**
	 * \brief 设置参数
	 * \param width				影像宽
	 * \param height			影像高
	 * \param min_disparity		最小视差
	 * \param max_disparity		最大视差
	 * \param p1				惩罚项P1
	 * \param p2				惩罚项P2
	 * \param tso				纹理阈值
	 */
	void SetParam(const sint32& width, const sint32& height, const sint32& min_disparity, const sint32& max_disparity,
		const float32& p1, const float32& p2, const sint32& tso);

	/**
	 * \brief 执行优化
	 */
	void Optimize();

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
	 * \brief 水平方向扫描线优化
	 * \param cost_so_src	输入代价数据
	 * \param cost_so_dst	输出代价数据
	 * \param is_forward	是否正向扫描
	 */
	void ScanlineOptimizeLeftRight(const float32* cost_so_src, float32* cost_so_dst, bool is_forward);

	/**
	 * \brief 垂直方向扫描线优化
	 * \param cost_so_src	输入代价数据
	 * \param cost_so_dst	输出代价数据  
	 * \param is_forward	是否正向扫描
	 */
	void ScanlineOptimizeUpDown(const float32* cost_so_src, float32* cost_so_dst, bool is_forward);

	/**
	 * \brief 计算视差图
	 */
	void ComputeDisparity();

	/**
	 * \brief 内联颜色距离计算
	 * \param c1 颜色1
	 * \param c2 颜色2
	 * \return 颜色距离
	 */
	inline uint8 ColorDist(const ADColor& c1, const ADColor& c2);

private:
	/** \brief 影像尺寸 */
	sint32 width_, height_;
	
	/** \brief 左右影像数据 */
	const uint8* img_left_;
	const uint8* img_right_;
	
	/** \brief 代价数据 */
	float32* cost_init_;
	float32* cost_aggr_;
	
	/** \brief 视差范围 */
	sint32 min_disparity_, max_disparity_;
	
	/** \brief 算法参数 */
	float32 so_p1_, so_p2_;
	sint32 so_tso_;

	/** \brief 视差图 */
	std::vector<float32> disp_left_;
	std::vector<float32> disp_right_;
	
	/** \brief 优化：预分配的缓冲区 */
	std::vector<float32> cost_last_path_;
	std::vector<float32> temp_cost_buffer_;
};

#endif