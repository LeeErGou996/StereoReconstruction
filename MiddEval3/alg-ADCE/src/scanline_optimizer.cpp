/* -*-c++-*- AD-Census - Copyright (C) 2020.
* Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
* https://github.com/ethan-li-coding/AD-Census
* Describe	: implement of class ScanlineOptimizer (Optimized Version)
*/

#include "scanline_optimizer.h"

#include <cassert>
#include <cstring>
#include <algorithm>
#include <limits>

ScanlineOptimizer::ScanlineOptimizer(): width_(0), height_(0), img_left_(nullptr), img_right_(nullptr),
                                        cost_init_(nullptr), cost_aggr_(nullptr),
                                        min_disparity_(0), max_disparity_(0),
                                        so_p1_(0), so_p2_(0),
                                        so_tso_(0) {
    // 预分配缓冲区，避免重复分配
    cost_last_path_.reserve(256);  // 预分配最大可能的视差范围
}

ScanlineOptimizer::~ScanlineOptimizer() {}

void ScanlineOptimizer::SetData(const uint8* img_left, const uint8* img_right, float32* cost_init,
	float32* cost_aggr)
{
	img_left_ = img_left;
	img_right_ = img_right;
	cost_init_ = cost_init;
	cost_aggr_ = cost_aggr;
}

void ScanlineOptimizer::SetData(const float32* cost_left, const float32* cost_right)
{
	// 修复：为ADCE版本处理空指针问题
	cost_init_ = const_cast<float32*>(cost_left);
	cost_aggr_ = const_cast<float32*>(cost_right);
	// 对于ADCE版本，图像数据可能不可用，设置标志
	img_left_ = nullptr;
	img_right_ = nullptr;
}

void ScanlineOptimizer::SetParam(const sint32& width, const sint32& height, const sint32& min_disparity,
	const sint32& max_disparity, const float32& p1, const float32& p2, const sint32& tso)
{
	width_ = width;
	height_ = height;
	min_disparity_ = min_disparity;
	max_disparity_ = max_disparity;
	so_p1_ = p1;
	so_p2_ = p2;
	so_tso_ = tso;
	
	// 预分配cost_last_path_缓冲区
	const sint32 disp_range = max_disparity - min_disparity;
	cost_last_path_.resize(disp_range + 2);
}

void ScanlineOptimizer::Optimize()
{
	// 修复：处理图像数据为空的情况
	if (width_ <= 0 || height_ <= 0 || cost_init_ == nullptr || cost_aggr_ == nullptr) {
		return;
	}
	
	// 分配临时缓冲区，避免输入输出数据冲突
	const sint32 cost_size = width_ * height_ * (max_disparity_ - min_disparity_);
	temp_cost_buffer_.resize(cost_size);
	
	// 4方向扫描线优化
	// 使用临时缓冲区避免数据竞争
	
	// left to right
	ScanlineOptimizeLeftRight(cost_aggr_, temp_cost_buffer_.data(), true);
	// right to left  
	ScanlineOptimizeLeftRight(temp_cost_buffer_.data(), cost_aggr_, false);
	// up to down
	ScanlineOptimizeUpDown(cost_aggr_, temp_cost_buffer_.data(), true);
	// down to up
	ScanlineOptimizeUpDown(temp_cost_buffer_.data(), cost_aggr_, false);
	
	// 计算视差图
	ComputeDisparity();
}

// 内联颜色距离计算，提高性能
inline uint8 ScanlineOptimizer::ColorDist(const ADColor& c1, const ADColor& c2) {
	const int dr = static_cast<int>(c1.r) - static_cast<int>(c2.r);
	const int dg = static_cast<int>(c1.g) - static_cast<int>(c2.g);
	const int db = static_cast<int>(c1.b) - static_cast<int>(c2.b);
	return static_cast<uint8>(std::max(std::max(abs(dr), abs(dg)), abs(db)));
}

void ScanlineOptimizer::ScanlineOptimizeLeftRight(const float32* cost_so_src, float32* cost_so_dst, bool is_forward)
{
	const auto width = width_;
	const auto height = height_;
	const auto min_disparity = min_disparity_;
	const auto max_disparity = max_disparity_;
	const auto p1 = so_p1_;
	const auto p2 = so_p2_;
	const auto tso = so_tso_;
	
	assert(width > 0 && height > 0 && max_disparity > min_disparity);

	// 视差范围
	const sint32 disp_range = max_disparity - min_disparity;

	// 正向(左->右) ：is_forward = true ; direction = 1
	// 反向(右->左) ：is_forward = false; direction = -1;
	const sint32 direction = is_forward ? 1 : -1;

	// 聚合
	for (sint32 y = 0; y < height; y++) {
		// 路径头为每一行的首(尾,dir=-1)像素
		auto cost_init_row = (is_forward) ? (cost_so_src + y * width * disp_range) : (cost_so_src + y * width * disp_range + (width - 1) * disp_range);
		auto cost_aggr_row = (is_forward) ? (cost_so_dst + y * width * disp_range) : (cost_so_dst + y * width * disp_range + (width - 1) * disp_range);
		
		sint32 x = (is_forward) ? 0 : width - 1;

		// 路径上的当前像素值、上一个像素值
		ADColor color, color_last;
		
		// 修复：处理图像数据为空的情况
		if (img_left_ != nullptr) {
			auto img_row = (is_forward) ? (img_left_ + y * width * 3) : (img_left_ + y * width * 3 + 3 * (width - 1));
			color = ADColor(img_row[0], img_row[1], img_row[2]);
			color_last = color;
		} else {
			// 如果没有图像数据，使用默认值
			color = ADColor(128, 128, 128);
			color_last = color;
		}

		// 使用预分配的缓冲区
		std::fill(cost_last_path_.begin(), cost_last_path_.end(), Large_Float);

		// 初始化：第一个像素的聚合代价值等于初始代价值
		memcpy(cost_aggr_row, cost_init_row, disp_range * sizeof(float32));
		memcpy(&cost_last_path_[1], cost_aggr_row, disp_range * sizeof(float32));
		cost_init_row += direction * disp_range;
		cost_aggr_row += direction * disp_range;
		x += direction;

		// 路径上第一个像素的最小代价值
		float32 mincost_last_path = *std::min_element(cost_last_path_.begin(), cost_last_path_.end());

		// 对方向上的第2个像素开始，顺序聚合
		for (sint32 j = 0; j < width - 1; j++) {
			// 获取当前像素颜色
			if (img_left_ != nullptr) {
				auto img_row = (is_forward) ? (img_left_ + y * width * 3 + x * 3) : (img_left_ + y * width * 3 + x * 3);
				color = ADColor(img_row[0], img_row[1], img_row[2]);
			}
			
			const uint8 d1 = (img_left_ != nullptr) ? ColorDist(color, color_last) : 0;
			uint8 d2 = d1;
			float32 min_cost = Large_Float;
			
			for (sint32 d = 0; d < disp_range; d++) {
				const sint32 xr = x - d - min_disparity;
				// 修复：边界检查
				if (img_right_ != nullptr && xr >= 0 && xr < width) {
					const sint32 xr_last = xr - direction;
					if (xr_last >= 0 && xr_last < width) {
						const auto img_row_r = img_right_ + y * width * 3;
						const ADColor color_r = ADColor(img_row_r[3 * xr], img_row_r[3 * xr + 1], img_row_r[3 * xr + 2]);
						const ADColor color_last_r = ADColor(img_row_r[3 * xr_last], img_row_r[3 * xr_last + 1], img_row_r[3 * xr_last + 2]);
						d2 = ColorDist(color_r, color_last_r);
					}
				}

				// 计算P1、P2
				float32 P1, P2;
				if (d1 < tso && d2 < tso) {
					P1 = p1; P2 = p2;
				}
				else if (d1 < tso && d2 >= tso) {
					P1 = p1 / 4; P2 = p2 / 4;
				}
				else if (d1 >= tso && d2 < tso) {
					P1 = p1 / 4; P2 = p2 / 4;
				}
				else {
					P1 = p1 / 10; P2 = p2 / 10;
				}

				// Lr(p,d) = C(p,d) + min( Lr(p-r,d), Lr(p-r,d-1) + P1, Lr(p-r,d+1) + P1, min(Lr(p-r))+P2 ) - min(Lr(p-r))
				const float32 cost = cost_init_row[d];
				const float32 l1 = cost_last_path_[d + 1];
				const float32 l2 = cost_last_path_[d] + P1;
				const float32 l3 = cost_last_path_[d + 2] + P1;
				const float32 l4 = mincost_last_path + P2;

				float32 cost_s = cost + std::min({l1, l2, l3, l4}) - mincost_last_path;

				cost_aggr_row[d] = cost_s;
				min_cost = std::min(min_cost, cost_s);
			}

			// 更新上个像素的最小代价值、代价数组
			mincost_last_path = min_cost;
			memcpy(&cost_last_path_[1], cost_aggr_row, disp_range * sizeof(float32));

			// 下一个像素
			cost_init_row += direction * disp_range;
			cost_aggr_row += direction * disp_range;
			x += direction;

			// 像素值更新
			color_last = color;
		}
	}
}

void ScanlineOptimizer::ScanlineOptimizeUpDown(const float32* cost_so_src, float32* cost_so_dst, bool is_forward)
{
	const auto width = width_;
	const auto height = height_;
	const auto min_disparity = min_disparity_;
	const auto max_disparity = max_disparity_;
	const auto p1 = so_p1_;
	const auto p2 = so_p2_;
	const auto tso = so_tso_;
	
	assert(width > 0 && height > 0 && max_disparity > min_disparity);

	// 视差范围
	const sint32 disp_range = max_disparity - min_disparity;

	// 正向(上->下) ：is_forward = true ; direction = 1
	// 反向(下->上) ：is_forward = false; direction = -1;
	const sint32 direction = is_forward ? 1 : -1;

	// 聚合
	for (sint32 x = 0; x < width; x++) {
		// 路径头为每一列的首(尾,dir=-1)像素
		auto cost_init_col = (is_forward) ? (cost_so_src + x * disp_range) : (cost_so_src + (height - 1) * width * disp_range + x * disp_range);
		auto cost_aggr_col = (is_forward) ? (cost_so_dst + x * disp_range) : (cost_so_dst + (height - 1) * width * disp_range + x * disp_range);
		
		sint32 y = (is_forward) ? 0 : height - 1;

		// 路径上的当前灰度值、上一个灰度值
		ADColor color, color_last;
		
		// 修复：处理图像数据为空的情况
		if (img_left_ != nullptr) {
			auto img_col = (is_forward) ? (img_left_ + 3 * x) : (img_left_ + (height - 1) * width * 3 + 3 * x);
			color = ADColor(img_col[0], img_col[1], img_col[2]);
			color_last = color;
		} else {
			color = ADColor(128, 128, 128);
			color_last = color;
		}

		// 使用预分配的缓冲区
		std::fill(cost_last_path_.begin(), cost_last_path_.end(), Large_Float);

		// 初始化：第一个像素的聚合代价值等于初始代价值
		memcpy(cost_aggr_col, cost_init_col, disp_range * sizeof(float32));
		memcpy(&cost_last_path_[1], cost_aggr_col, disp_range * sizeof(float32));
		cost_init_col += direction * width * disp_range;
		cost_aggr_col += direction * width * disp_range;
		y += direction;

		// 路径上第一个像素的最小代价值
		float32 mincost_last_path = *std::min_element(cost_last_path_.begin(), cost_last_path_.end());

		// 对方向上的第2个像素开始，顺序聚合
		for (sint32 i = 0; i < height - 1; i++) {
			// 获取当前像素颜色
			if (img_left_ != nullptr) {
				auto img_col = img_left_ + y * width * 3 + 3 * x;
				color = ADColor(img_col[0], img_col[1], img_col[2]);
			}
			
			const uint8 d1 = (img_left_ != nullptr) ? ColorDist(color, color_last) : 0;
			uint8 d2 = d1;
			float32 min_cost = Large_Float;
			
			for (sint32 d = 0; d < disp_range; d++) {
				const sint32 xr = x - d - min_disparity;
				// 修复：边界检查
				if (img_right_ != nullptr && xr >= 0 && xr < width) {
					const sint32 yr_last = y - direction;
					if (yr_last >= 0 && yr_last < height) {
						const ADColor color_r = ADColor(img_right_[y * width * 3 + 3 * xr], 
						                               img_right_[y * width * 3 + 3 * xr + 1], 
						                               img_right_[y * width * 3 + 3 * xr + 2]);
						const ADColor color_last_r = ADColor(img_right_[yr_last * width * 3 + 3 * xr],
						                                    img_right_[yr_last * width * 3 + 3 * xr + 1],
						                                    img_right_[yr_last * width * 3 + 3 * xr + 2]);
						d2 = ColorDist(color_r, color_last_r);
					}
				}
				
				// 计算P1、P2
				float32 P1, P2;
				if (d1 < tso && d2 < tso) {
					P1 = p1; P2 = p2;
				}
				else if (d1 < tso && d2 >= tso) {
					P1 = p1 / 4; P2 = p2 / 4;
				}
				else if (d1 >= tso && d2 < tso) {
					P1 = p1 / 4; P2 = p2 / 4;
				}
				else {
					P1 = p1 / 10; P2 = p2 / 10;
				}

				// Lr(p,d) = C(p,d) + min( Lr(p-r,d), Lr(p-r,d-1) + P1, Lr(p-r,d+1) + P1, min(Lr(p-r))+P2 ) - min(Lr(p-r))
				const float32 cost = cost_init_col[d];
				const float32 l1 = cost_last_path_[d + 1];
				const float32 l2 = cost_last_path_[d] + P1;
				const float32 l3 = cost_last_path_[d + 2] + P1;
				const float32 l4 = mincost_last_path + P2;

				float32 cost_s = cost + std::min({l1, l2, l3, l4}) - mincost_last_path;

				cost_aggr_col[d] = cost_s;
				min_cost = std::min(min_cost, cost_s);
			}

			// 更新上个像素的最小代价值、代价数组
			mincost_last_path = min_cost;
			memcpy(&cost_last_path_[1], cost_aggr_col, disp_range * sizeof(float32));

			// 下一个像素
			cost_init_col += direction * width * disp_range;
			cost_aggr_col += direction * width * disp_range;
			y += direction;

			// 像素值更新
			color_last = color;
		}
	}
}

const float32* ScanlineOptimizer::GetDisparityLeft() const
{
	return disp_left_.empty() ? nullptr : &disp_left_[0];
}

const float32* ScanlineOptimizer::GetDisparityRight() const
{
	return disp_right_.empty() ? nullptr : &disp_right_[0];
}

void ScanlineOptimizer::ComputeDisparity()
{
	const sint32 disp_range = max_disparity_ - min_disparity_;
	if (disp_range <= 0) {
		return;
	}
	
	const sint32 img_size = width_ * height_;
	
	// 初始化视差图
	disp_left_.resize(img_size);
	disp_right_.resize(img_size);
	
	// 优化：使用更高效的WTA算法
	const float32* cost_ptr = cost_aggr_;
	
	// 计算左视差图
	for (sint32 y = 0; y < height_; y++) {
		for (sint32 x = 0; x < width_; x++) {
			const sint32 pixel_idx = y * width_ + x;
			const float32* pixel_cost = cost_ptr + pixel_idx * disp_range;
			
			// 找到最小代价及其索引
			const float32* min_cost_ptr = std::min_element(pixel_cost, pixel_cost + disp_range);
			const sint32 best_disp_idx = static_cast<sint32>(min_cost_ptr - pixel_cost);
			
			// 亚像素精度优化
			float32 disparity = static_cast<float32>(min_disparity_ + best_disp_idx);
			
			// 边界检查后进行亚像素优化
			if (best_disp_idx > 0 && best_disp_idx < disp_range - 1) {
				const float32 c1 = pixel_cost[best_disp_idx - 1];
				const float32 c2 = pixel_cost[best_disp_idx];
				const float32 c3 = pixel_cost[best_disp_idx + 1];
				const float32 denom = c1 + c3 - 2 * c2;
				if (std::abs(denom) > 1e-6f) {
					disparity += (c1 - c3) / (2 * denom);
				}
			}
			
			disp_left_[pixel_idx] = disparity;
		}
	}
	
	// 修复：正确计算右视差图
	// 右视差图需要考虑视差的对称性
	for (sint32 y = 0; y < height_; y++) {
		for (sint32 x = 0; x < width_; x++) {
			const sint32 pixel_idx = y * width_ + x;
			float32 min_cost = std::numeric_limits<float32>::max();
			sint32 best_disparity = min_disparity_;
			
			// 对于右图像的每个像素，在左图像中寻找最佳匹配
			for (sint32 d = 0; d < disp_range; d++) {
				const sint32 xl = x + min_disparity_ + d;
				if (xl >= 0 && xl < width_) {
					const float32 cost = cost_ptr[(y * width_ + xl) * disp_range + d];
					if (cost < min_cost) {
						min_cost = cost;
						best_disparity = min_disparity_ + d;
					}
				}
			}
			
			disp_right_[pixel_idx] = static_cast<float32>(best_disparity);
		}
	}
}