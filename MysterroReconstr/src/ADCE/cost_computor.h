/* -*-c++-*- AD-Census - Copyright (C) 2020.
* Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
* https://github.com/ethan-li-coding/AD-Census
* Describe	: header of class CostComputor (Optimized Version)
*/

#ifndef ADCENSUS_COST_COMPUTOR_H_
#define ADCENSUS_COST_COMPUTOR_H_

#include "adcensus_types.h"
#include <vector>

/**
 * \brief CostComputor类，用于计算AD-Census代价
 */
class CostComputor
{
private:
	// exp查找表常量
	static constexpr sint32 EXP_TABLE_SIZE = 1024;
	static constexpr float32 MAX_EXP_INPUT = 10.0f;

public:
	CostComputor();
	~CostComputor();

	/**
	 * \brief 初始化
	 * \param width				影像宽度
	 * \param height			影像高度
	 * \param min_disparity		最小视差
	 * \param max_disparity		最大视差
	 * \return true:成功 false:失败
	 */
	bool Initialize(const sint32& width, const sint32& height, const sint32& min_disparity, const sint32& max_disparity);

	/**
	 * \brief 设置影像数据
	 * \param img_left		左影像数据，三通道
	 * \param img_right		右影像数据，三通道
	 */
	void SetData(const uint8* img_left, const uint8* img_right);

	/**
	 * \brief 设置参数
	 * \param lambda_ad			AD代价权重参数
	 * \param lambda_census		Census代价权重参数
	 */
	void SetParams(const float32& lambda_ad, const float32& lambda_census);

	/**
	 * \brief 执行代价计算
	 */
	void Compute();

	/**
	 * \brief 获取代价数据指针（向后兼容）
	 * \return 代价数据指针
	 */
	float32* get_cost_ptr();

	/**
	 * \brief 获取左影像代价数据
	 * \return 左影像代价数据指针
	 */
	const float32* GetCostLeft() const;

	/**
	 * \brief 获取右影像代价数据
	 * \return 右影像代价数据指针
	 */
	const float32* GetCostRight() const;

private:
	/**
	 * \brief 计算灰度图
	 */
	void ComputeGray();

	/**
	 * \brief Census变换
	 */
	void CensusTransform();

	/**
	 * \brief 9x7窗口Census变换
	 * \param gray		灰度图
	 * \param census	Census数据
	 * \param width		影像宽度
	 * \param height	影像高度
	 */
	void CensusTransform9x7(const uint8* gray, uint64* census, const sint32 width, const sint32 height);

	/**
	 * \brief 计算代价
	 */
	void ComputeCost();

	/**
	 * \brief 计算64位Hamming距离
	 * \param a		第一个数值
	 * \param b		第二个数值
	 * \return Hamming距离
	 */
	static uint32 Hamming64(const uint64 a, const uint64 b);

	/**
	 * \brief 初始化exp查找表
	 */
	void InitExpLookupTable();

	/**
	 * \brief 从查找表获取exp值
	 * \param x		输入值
	 * \return exp(-x)的近似值
	 */
	float32 GetExpValue(const float32 x) const;

private:
	/** \brief 影像尺寸 */
	sint32 width_, height_;

	/** \brief 影像数据指针 */
	const uint8* img_left_;
	const uint8* img_right_;

	/** \brief 算法参数 */
	float32 lambda_ad_;
	float32 lambda_census_;

	/** \brief 视差范围 */
	sint32 min_disparity_, max_disparity_;

	/** \brief 初始化标志 */
	bool is_initialized_;

	/** \brief 灰度数据 */
	std::vector<uint8> gray_left_;
	std::vector<uint8> gray_right_;

	/** \brief Census数据 */
	std::vector<uint64> census_left_;
	std::vector<uint64> census_right_;

	/** \brief 代价数据 */
	std::vector<float32> cost_left_;		// 左影像代价
	std::vector<float32> cost_right_;		// 右影像代价
	std::vector<float32> cost_init_;		// 向后兼容的代价数组

	/** \brief exp函数查找表 */
	std::vector<float32> exp_table_;
};

#endif // ADCENSUS_COST_COMPUTOR_H_