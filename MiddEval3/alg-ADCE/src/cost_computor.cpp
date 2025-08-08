/* -*-c++-*- AD-Census - Copyright (C) 2020.
* Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
* https://github.com/ethan-li-coding/AD-Census
* Describe	: implement of class CostComputor (Optimized Version)
*/

#include "cost_computor.h"
#include "adcensus_types.h"
#include <cmath>
#include <algorithm>
#include <cstring>

CostComputor::CostComputor(): width_(0), height_(0), img_left_(nullptr), img_right_(nullptr),
                              lambda_ad_(10.0f), lambda_census_(30.0f), min_disparity_(0), max_disparity_(0),
                              is_initialized_(false) { 
    // 预计算exp查找表，提高性能
    InitExpLookupTable();
}

CostComputor::~CostComputor()
{
	
}

bool CostComputor::Initialize(const sint32& width, const sint32& height, const sint32& min_disparity, const sint32& max_disparity)
{
	width_ = width;
	height_ = height;
	min_disparity_ = min_disparity;
	max_disparity_ = max_disparity;

	const sint32 img_size = width_ * height_;
	const sint32 disp_range = max_disparity_ - min_disparity_;
	if (img_size <= 0 || disp_range <= 0) {
		is_initialized_ = false;
		return false;
	}

	try {
		// 灰度数据：左右影像
	gray_left_.resize(img_size);
	gray_right_.resize(img_size);
		
		// census数据：左右影像
		census_left_.resize(img_size, 0);
		census_right_.resize(img_size, 0);
		
		// 修复：分别存储左右代价数据
		cost_left_.resize(img_size * disp_range);
		cost_right_.resize(img_size * disp_range);
		
		// 保持向后兼容
	cost_init_.resize(img_size * disp_range);

		is_initialized_ = true;
	}
	catch (const std::bad_alloc&) {
		is_initialized_ = false;
	}
	
	return is_initialized_;
}

void CostComputor::SetData(const uint8* img_left, const uint8* img_right)
{
	img_left_ = img_left;
	img_right_ = img_right;
}

void CostComputor::SetParams(const float32& lambda_ad, const float32& lambda_census)
{
	lambda_ad_ = lambda_ad;
	lambda_census_ = lambda_census;
}

void CostComputor::ComputeGray()
{
	// 优化：并行化的RGB到灰度转换
	#pragma omp parallel for
	for (sint32 n = 0; n < 2; n++) {
		const auto color = (n == 0) ? img_left_ : img_right_;
		auto& gray = (n == 0) ? gray_left_ : gray_right_;
		
		for (sint32 y = 0; y < height_; y++) {
			const uint8* color_row = color + y * width_ * 3;
			uint8* gray_row = &gray[y * width_];
			
			for (sint32 x = 0; x < width_; x++) {
				const uint8* pixel = color_row + 3 * x;
				const uint8 b = pixel[0];
				const uint8 g = pixel[1];
				const uint8 r = pixel[2];
				// 使用整数运算优化灰度转换
				gray_row[x] = static_cast<uint8>((r * 77 + g * 151 + b * 28) >> 8);
			}
		}
	}
}

void CostComputor::CensusTransform()
{
	// 内部实现9x7 Census变换
	CensusTransform9x7(&gray_left_[0], &census_left_[0], width_, height_);
	CensusTransform9x7(&gray_right_[0], &census_right_[0], width_, height_);
}

void CostComputor::CensusTransform9x7(const uint8* gray, uint64* census, const sint32 width, const sint32 height)
{
	const sint32 radius_x = 4;  // 9/2
	const sint32 radius_y = 3;  // 7/2
	
	#pragma omp parallel for
	for (sint32 y = radius_y; y < height - radius_y; y++) {
		for (sint32 x = radius_x; x < width - radius_x; x++) {
			const uint8 center = gray[y * width + x];
			uint64 census_val = 0;
			
			// 7x9窗口内的Census变换
			for (sint32 dy = -radius_y; dy <= radius_y; dy++) {
				for (sint32 dx = -radius_x; dx <= radius_x; dx++) {
					if (dx == 0 && dy == 0) continue;  // 跳过中心像素
					
					const uint8 neighbor = gray[(y + dy) * width + (x + dx)];
					census_val <<= 1;
					if (neighbor < center) {
						census_val |= 1;
					}
				}
			}
			census[y * width + x] = census_val;
		}
	}
}

uint32 CostComputor::Hamming64(const uint64 a, const uint64 b)
{
	// 使用内置函数计算Hamming距离（如果可用）
	#ifdef __GNUC__
	return __builtin_popcountll(a ^ b);
	#else
	// 手动实现
	uint64 diff = a ^ b;
	uint32 count = 0;
	while (diff) {
		count += diff & 1;
		diff >>= 1;
	}
	return count;
	#endif
}

void CostComputor::ComputeCost()
{
	const sint32 disp_range = max_disparity_ - min_disparity_;

	// 预计算常量
	const float32 inv_lambda_ad = 1.0f / lambda_ad_;
	const float32 inv_lambda_census = 1.0f / lambda_census_;
	
	// 优化：使用更缓存友好的访问模式
	#pragma omp parallel for
	for (sint32 y = 0; y < height_; y++) {
		// 预计算行偏移
		const sint32 row_offset = y * width_;
		const sint32 img_row_offset = y * width_ * 3;
		const sint32 cost_row_offset = y * width_ * disp_range;
		
		const uint8* img_left_row = img_left_ + img_row_offset;
		const uint8* img_right_row = img_right_ + img_row_offset;
		const uint64* census_left_row = &census_left_[row_offset];
		const uint64* census_right_row = &census_right_[row_offset];
		
		for (sint32 x = 0; x < width_; x++) {
			// 预计算像素偏移
			const sint32 pixel_offset = 3 * x;
			const uint8 bl = img_left_row[pixel_offset];
			const uint8 gl = img_left_row[pixel_offset + 1];
			const uint8 rl = img_left_row[pixel_offset + 2];
			const uint64 census_val_l = census_left_row[x];
			
			// 计算代价数组的基地址
			float32* cost_left_base = &cost_left_[cost_row_offset + x * disp_range];
			float32* cost_right_base = &cost_right_[cost_row_offset + x * disp_range];
			float32* cost_init_base = &cost_init_[cost_row_offset + x * disp_range];
			
			// 遍历视差范围
			for (sint32 d = 0; d < disp_range; d++) {
				const sint32 disp = min_disparity_ + d;
				const sint32 xr = x - disp;
				
				float32 cost_val;
				
				if (xr < 0 || xr >= width_) {
					// 修复：使用更合理的边界代价
					cost_val = 2.0f;  // 最大代价值
				} else {
					// AD代价计算
					const sint32 xr_pixel_offset = 3 * xr;
					const uint8 br = img_right_row[xr_pixel_offset];
					const uint8 gr = img_right_row[xr_pixel_offset + 1];
					const uint8 rr = img_right_row[xr_pixel_offset + 2];
					
					const float32 cost_ad = (std::abs(static_cast<sint32>(bl) - static_cast<sint32>(br)) +
					                        std::abs(static_cast<sint32>(gl) - static_cast<sint32>(gr)) +
					                        std::abs(static_cast<sint32>(rl) - static_cast<sint32>(rr))) / 3.0f;
					
					// Census代价计算
					const uint64 census_val_r = census_right_row[xr];
					const float32 cost_census = static_cast<float32>(Hamming64(census_val_l, census_val_r));
					
					// 优化：使用查找表计算exp函数
					const float32 exp_ad = GetExpValue(cost_ad * inv_lambda_ad);
					const float32 exp_census = GetExpValue(cost_census * inv_lambda_census);
					
					// AD-Census代价公式
					cost_val = (1.0f - exp_ad) + (1.0f - exp_census);
				}
				
				// 存储到所有相关数组
				cost_left_base[d] = cost_val;
				cost_right_base[d] = cost_val;  // 对于标准实现，左右代价相同
				cost_init_base[d] = cost_val;   // 向后兼容
			}
		}
	}
}

void CostComputor::Compute()
{
	if(!is_initialized_ || img_left_ == nullptr || img_right_ == nullptr) {
		return;
	}

	// 计算灰度图
	ComputeGray();

	// census变换
	CensusTransform();

	// 代价计算
	ComputeCost();
}

void CostComputor::InitExpLookupTable()
{
	// 初始化exp函数查找表，范围[0, MAX_EXP_INPUT]
	const float32 step = MAX_EXP_INPUT / EXP_TABLE_SIZE;
	exp_table_.resize(EXP_TABLE_SIZE + 1);
	
	for (sint32 i = 0; i <= EXP_TABLE_SIZE; i++) {
		const float32 x = i * step;
		exp_table_[i] = std::exp(-x);
	}
}

float32 CostComputor::GetExpValue(const float32 x) const
{
	if (x >= MAX_EXP_INPUT) {
		return 0.0f;
	}
	if (x <= 0.0f) {
		return 1.0f;
	}
	
	// 线性插值查找表
	const float32 index_f = x * EXP_TABLE_SIZE / MAX_EXP_INPUT;
	const sint32 index = static_cast<sint32>(index_f);
	const float32 frac = index_f - index;
	
	if (index >= EXP_TABLE_SIZE) {
		return exp_table_[EXP_TABLE_SIZE];
	}
	
	return exp_table_[index] * (1.0f - frac) + exp_table_[index + 1] * frac;
}

float32* CostComputor::get_cost_ptr()
{
	return cost_init_.empty() ? nullptr : &cost_init_[0];
}

const float32* CostComputor::GetCostLeft() const
{
	return cost_left_.empty() ? nullptr : &cost_left_[0];
}

const float32* CostComputor::GetCostRight() const
{
	return cost_right_.empty() ? nullptr : &cost_right_[0];
}