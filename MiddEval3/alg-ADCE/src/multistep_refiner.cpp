/* -*-c++-*- AD-Census - Copyright (C) 2020.
* Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
* https://github.com/ethan-li-coding/AD-Census
* Describe	: implement of class MultiStepRefiner (Optimized Version)
*/

#include "multistep_refiner.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>

MultiStepRefiner::MultiStepRefiner(): width_(0), height_(0), img_left_(nullptr), cost_(nullptr),
                                      cross_arms_(nullptr),
                                      disp_left_(nullptr), disp_right_(nullptr),
                                      min_disparity_(0), max_disparity_(0),
                                      irv_ts_(0), irv_th_(0), lrcheck_thres_(0),
                                      do_lr_check_(false), do_region_voting_(false),
                                      do_interpolating_(false), do_discontinuity_adjustment_(false),
                                      is_adce_mode_(false) { 
    // 预分配缓冲区
    disp_collects_.reserve(32);
}

MultiStepRefiner::~MultiStepRefiner()
{
}

bool MultiStepRefiner::Initialize(const sint32& width, const sint32& height)
{
	width_ = width;
	height_ = height;
	if (width_ <= 0 || height_ <= 0) {
		return false;
	}

	const sint32 img_size = width * height;
	
	// 初始化边缘标记
	vec_edge_left_.clear();
	vec_edge_left_.resize(img_size, 0);
	
	// 预分配其他缓冲区
	temp_histogram_.resize(256);  // 预分配直方图缓冲区
	
	return true;
}

void MultiStepRefiner::SetData(const uint8* img_left, float32* cost, const CrossArm* cross_arms, 
                               float32* disp_left, float32* disp_right)
{
	img_left_ = img_left;
	cost_ = cost; 
	cross_arms_ = cross_arms;
	disp_left_ = disp_left;
	disp_right_ = disp_right;
	is_adce_mode_ = false;  // 标准模式
}

void MultiStepRefiner::SetData(const float32* disp_left, const float32* disp_right)
{
	// 修复：ADCE模式处理
	disp_left_ = const_cast<float32*>(disp_left);
	disp_right_ = const_cast<float32*>(disp_right);
	// 设置ADCE模式标志
	is_adce_mode_ = true;
	// ADCE模式下，其他数据不可用
	img_left_ = nullptr;
	cost_ = nullptr;
	cross_arms_ = nullptr;
}

void MultiStepRefiner::SetParam(const sint32& min_disparity, const sint32& max_disparity, 
                                const sint32& irv_ts, const float32& irv_th, const float32& lrcheck_thres,
								const bool& do_lr_check, const bool& do_region_voting, 
								const bool& do_interpolating, const bool& do_discontinuity_adjustment)
{
	min_disparity_ = min_disparity;
	max_disparity_ = max_disparity;
	irv_ts_ = irv_ts;
	irv_th_ = irv_th;
	lrcheck_thres_ = lrcheck_thres;
	do_lr_check_ = do_lr_check;
	do_region_voting_ = do_region_voting;
	do_interpolating_ = do_interpolating;
	do_discontinuity_adjustment_ = do_discontinuity_adjustment;
}

void MultiStepRefiner::Refine()
{
	// 修复：根据模式检查必要的数据
	if (width_ <= 0 || height_ <= 0 || disp_left_ == nullptr || disp_right_ == nullptr) {
		return;
	}
	
	// ADCE模式下，只执行基本的优化步骤
	if (is_adce_mode_) {
		// step1: outlier detection (基于左右一致性检查)
		if (do_lr_check_) {
			OutlierDetection();
		}
		// step3: 简化的插值 (不依赖图像数据)
		if (do_interpolating_) {
			SimpleInterpolation();
		}
		// 简化的中值滤波
		SimpleMedianFilter();
		return;
	}
	
	// 标准模式：检查所有必要数据
	if (cost_ == nullptr || cross_arms_ == nullptr) {
		return;
	}

	// step1: outlier detection
	if (do_lr_check_) {
		OutlierDetection();
	}
	// step2: iterative region voting
	if (do_region_voting_) {
		IterativeRegionVoting();
	}
	// step3: proper interpolation
	if (do_interpolating_) {
		ProperInterpolation();
	}
	// step4: discontinuities adjustment
	if (do_discontinuity_adjustment_) {
		DepthDiscontinuityAdjustment();
	}

	// median filter
	MedianFilter();
}

void MultiStepRefiner::OutlierDetection()
{
	const sint32 width = width_;
	const sint32 height = height_;
	const float32& threshold = lrcheck_thres_;

	// 清空遮挡和误匹配像素列表
	occlusions_.clear();
	mismatches_.clear();

	// 左右一致性检查
	for (sint32 y = 0; y < height; y++) {
		for (sint32 x = 0; x < width; x++) {
			// 左影像视差值
			auto& disp = disp_left_[y * width + x];
			if (disp == Invalid_Float) {
				mismatches_.emplace_back(x, y);
				continue;
			}

			// 通过视差值找到右影像上对应的同名像素
			const auto col_right = lround(x - disp);
			if (col_right >= 0 && col_right < width) {
				// 右影像同名像素的视差值
				const auto& disp_r = disp_right_[y * width + col_right];
				// 判断左右视差值是否一致（差值在阈值内）
				if (std::abs(disp - disp_r) > threshold) {
					// 进一步区分遮挡像素和误匹配像素
					const sint32 col_rl = lround(col_right + disp_r);
					if (col_rl >= 0 && col_rl < width) {  // 修复：边界检查
						const auto& disp_l = disp_left_[y * width + col_rl];
						if (disp_l > disp) {
							occlusions_.emplace_back(x, y);
						}
						else {
							mismatches_.emplace_back(x, y);
						}
					}
					else {
						mismatches_.emplace_back(x, y);
					}

					// 将视差值标记为无效
					disp = Invalid_Float;
				}
			}
			else {
				// 通过视差值在右影像上找不到同名像素，超出影像范围
				disp = Invalid_Float;
				mismatches_.emplace_back(x, y);
			}
		}
	}
}

void MultiStepRefiner::IterativeRegionVoting()
{
	// ADCE模式下跳过此函数，因为cross_arms_为nullptr
	if (is_adce_mode_ || cross_arms_ == nullptr) {
		return;
	}
	
	const sint32 width = width_;
	const auto disp_range = max_disparity_ - min_disparity_;
	if(disp_range <= 0) {
		return;
	}
	
	// 使用预分配的直方图缓冲区
	temp_histogram_.resize(disp_range);

	// 迭代5次，添加收敛检查
	const sint32 max_iters = 5;
	
	for (sint32 it = 0; it < max_iters; it++) {
		bool has_changes = false;
		
		for (sint32 k = 0; k < 2; k++) {
			auto& trg_pixels = (k == 0) ? mismatches_ : occlusions_;
			
			for (auto& pix : trg_pixels) {
				const sint32& x = pix.first;
				const sint32& y = pix.second;
				auto& disp = disp_left_[y * width + x];
				if(disp != Invalid_Float) {
					continue;
				}

				// 初始化直方图
				std::fill(temp_histogram_.begin(), temp_histogram_.begin() + disp_range, 0);

				// 修复：添加边界检查的支撑窗口统计，类型改为uint8
				const auto& arm = cross_arms_[y * width + x];
				
				// 检查支撑窗口边界（注意uint8类型转换）
				const sint32 top_bound = std::max(0, y - static_cast<sint32>(arm.top));
				const sint32 bottom_bound = std::min(height_ - 1, y + static_cast<sint32>(arm.bottom));
				
				for (sint32 yt = top_bound; yt <= bottom_bound; yt++) {
					const auto& arm2 = cross_arms_[yt * width + x];
					const sint32 left_bound = std::max(0, x - static_cast<sint32>(arm2.left));
					const sint32 right_bound = std::min(width - 1, x + static_cast<sint32>(arm2.right));
					
					for (sint32 xt = left_bound; xt <= right_bound; xt++) {
						const auto& d = disp_left_[yt * width + xt];
						if (d != Invalid_Float) {
							const auto di = lround(d);
							if (di >= min_disparity_ && di < max_disparity_) {
								temp_histogram_[di - min_disparity_]++;
							}
						}
					}
				}
				
				// 寻找直方图峰值对应的视差
				sint32 best_disp = 0, count = 0;
				sint32 max_ht = 0;
				for (sint32 d = 0; d < disp_range; d++) {
					const auto& h = temp_histogram_[d];
					if (max_ht < h) {
						max_ht = h;
						best_disp = d;
					}
					count += h;
				}

				if (max_ht > 0 && count > irv_ts_) {
					const float32 confidence = static_cast<float32>(max_ht) / count;
					if (confidence > irv_th_) {
						disp = best_disp + min_disparity_;
						has_changes = true;
					}
				}
			}
			
			// 删除已填充的像素
			trg_pixels.erase(
				std::remove_if(trg_pixels.begin(), trg_pixels.end(),
					[this, width](const std::pair<sint32, sint32>& pix) {
						return disp_left_[pix.second * width + pix.first] != Invalid_Float;
					}),
				trg_pixels.end());
		}
		
		// 如果没有变化，提前退出
		if (!has_changes) break;
	}
}

void MultiStepRefiner::ProperInterpolation()
{
	// ADCE模式下跳过此函数，因为需要图像数据
	if (is_adce_mode_ || img_left_ == nullptr) {
		return;
	}
	
	const sint32 width = width_;
	const sint32 height = height_;

	const float32 pi = 3.1415926f;
	const sint32 max_search_length = std::min(64, std::max(abs(max_disparity_), abs(min_disparity_)));

	std::vector<float32> fill_disps;
	
	for (sint32 k = 0; k < 2; k++) {
		auto& trg_pixels = (k == 0) ? mismatches_ : occlusions_;
		if (trg_pixels.empty()) {
			continue;
		}
		fill_disps.resize(trg_pixels.size());

		// 对每个待插值像素进行处理
		for (size_t n = 0; n < trg_pixels.size(); n++) {
			auto& pix = trg_pixels[n];
			const sint32 x = pix.first;
			const sint32 y = pix.second;

			// 修复：清空收集器并16方向搜索有效视差值
			disp_collects_.clear();
			
			for (sint32 s = 0; s < 16; s++) {
				const float32 ang = s * pi / 16;
				const float32 sina = std::sin(ang);
				const float32 cosa = std::cos(ang);
				
				for (sint32 m = 1; m < max_search_length; m++) {
					const sint32 yy = lround(y + m * sina);
					const sint32 xx = lround(x + m * cosa);
					if (yy < 0 || yy >= height || xx < 0 || xx >= width) { 
						break;
					}
					const auto& d = disp_left_[yy * width + xx];
					if (d != Invalid_Float) {
						// 修复：正确存储像素位置和视差值
						disp_collects_.emplace_back(yy * width + xx, d);
						break;
					}
				}
			}
			
			if (disp_collects_.empty()) {
				fill_disps[n] = 0.0f;  // 默认值
				continue;
			}

			// 根据像素类型选择插值策略
			if (k == 0) {  // 误匹配像素：选择颜色最相似的
				sint32 min_dist = 9999;
				float32 best_disp = 0.0f;
				const auto color = ADColor(img_left_[y * width * 3 + 3 * x], 
				                          img_left_[y * width * 3 + 3 * x + 1], 
				                          img_left_[y * width * 3 + 3 * x + 2]);
				
				for (const auto& dc : disp_collects_) {
					const sint32 pixel_idx = dc.first;
					const sint32 img_idx = (pixel_idx / width) * width * 3 + (pixel_idx % width) * 3;
					const auto color2 = ADColor(img_left_[img_idx], 
					                           img_left_[img_idx + 1], 
					                           img_left_[img_idx + 2]);
					const auto dist = ColorDist(color, color2);
					if (min_dist > dist) {
						min_dist = dist;
						best_disp = dc.second;
					}
				}
				fill_disps[n] = best_disp;
			}
			else {  // 遮挡像素：选择最小视差
				float32 min_disp = Large_Float;
				for (const auto& dc : disp_collects_) {
					min_disp = std::min(min_disp, dc.second);
				}
				fill_disps[n] = min_disp;
			}
		}
		
		// 应用插值结果
		for (size_t n = 0; n < trg_pixels.size(); n++) {
			auto& pix = trg_pixels[n];
			const sint32 x = pix.first;
			const sint32 y = pix.second;
			disp_left_[y * width + x] = fill_disps[n];
		}
	}
}

void MultiStepRefiner::SimpleInterpolation()
{
	// ADCE模式下的简化插值
	const sint32 width = width_;
	const sint32 height = height_;
	
	// 简单的邻域插值
	for (sint32 y = 1; y < height - 1; y++) {
		for (sint32 x = 1; x < width - 1; x++) {
			auto& disp = disp_left_[y * width + x];
			if (disp == Invalid_Float) {
				// 使用4邻域的平均值
				float32 sum = 0.0f;
				sint32 count = 0;
				
				// 检查4个邻域
				const sint32 offsets[] = {-width, width, -1, 1};
				for (sint32 offset : offsets) {
					const float32 neighbor_disp = disp_left_[y * width + x + offset];
					if (neighbor_disp != Invalid_Float) {
						sum += neighbor_disp;
						count++;
					}
				}
				
				if (count > 0) {
					disp = sum / count;
				}
			}
		}
	}
}

void MultiStepRefiner::DepthDiscontinuityAdjustment()
{
	// ADCE模式或cost_为空时跳过
	if (is_adce_mode_ || cost_ == nullptr) {
		return;
	}
	
	const sint32 width = width_;
	const sint32 height = height_;
	const auto disp_range = max_disparity_ - min_disparity_;
	if (disp_range <= 0) {
		return;
	}
	
	// 对视差图进行边缘检测
	const float32 edge_thres = 5.0f;
	EdgeDetect(&vec_edge_left_[0], disp_left_, width, height, edge_thres);

	// 调整边缘像素的视差
	for (sint32 y = 0; y < height; y++) {
		for (sint32 x = 1; x < width - 1; x++) {
			const auto& e_label = vec_edge_left_[y * width + x];
			if (e_label == 1) {
				const auto disp_ptr = disp_left_ + y * width;
				float32& d = disp_ptr[x];
				if (d != Invalid_Float) {
					const sint32 di = lround(d) - min_disparity_;
					if (di >= 0 && di < disp_range) {
						const auto cost_ptr = cost_ + y * width * disp_range + x * disp_range;
						float32 c0 = cost_ptr[di];

						// 比较左右邻域像素的代价
						for (int k = 0; k < 2; k++) {
							const sint32 x2 = (k == 0) ? x - 1 : x + 1;
							const float32& d2 = disp_ptr[x2];
							if (d2 != Invalid_Float) {
								const sint32 d2i = lround(d2) - min_disparity_;
								if (d2i >= 0 && d2i < disp_range) {
									// 修复：正确的代价索引计算
									const sint32 cost_offset = (k == 0) ? -disp_range : disp_range;
									const auto& c = cost_ptr[cost_offset + d2i];
									if (c < c0) {
										d = d2;
										c0 = c;
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

void MultiStepRefiner::EdgeDetect(uint8* edge_mask, const float32* disp_ptr, 
                                  const sint32& width, const sint32& height, const float32 threshold)
{
	memset(edge_mask, 0, width * height * sizeof(uint8));
	
	// 修复：添加无效值检查的Sobel边缘检测
	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			// 检查所有邻域像素是否有效
			bool all_valid = true;
			for (int dy = -1; dy <= 1 && all_valid; dy++) {
				for (int dx = -1; dx <= 1 && all_valid; dx++) {
					if (disp_ptr[(y + dy) * width + x + dx] == Invalid_Float) {
						all_valid = false;
					}
				}
			}
			
			if (all_valid) {
				const auto grad_x = (-disp_ptr[(y - 1) * width + x - 1] + disp_ptr[(y - 1) * width + x + 1]) +
					(-2 * disp_ptr[y * width + x - 1] + 2 * disp_ptr[y * width + x + 1]) +
					(-disp_ptr[(y + 1) * width + x - 1] + disp_ptr[(y + 1) * width + x + 1]);
				const auto grad_y = (-disp_ptr[(y - 1) * width + x - 1] - 2 * disp_ptr[(y - 1) * width + x] - disp_ptr[(y - 1) * width + x + 1]) +
					(disp_ptr[(y + 1) * width + x - 1] + 2 * disp_ptr[(y + 1) * width + x] + disp_ptr[(y + 1) * width + x + 1]);
				const auto grad = std::abs(grad_x) + std::abs(grad_y);
				if (grad > threshold) {
					edge_mask[y * width + x] = 1;
				}
			}
		}
	}
}

void MultiStepRefiner::MedianFilter()
{
	// 实现3x3中值滤波
	const sint32 width = width_;
	const sint32 height = height_;
	
	std::vector<float32> temp_disp(width * height);
	std::copy(disp_left_, disp_left_ + width * height, temp_disp.begin());
	
	for (sint32 y = 1; y < height - 1; y++) {
		for (sint32 x = 1; x < width - 1; x++) {
			std::vector<float32> neighbors;
			neighbors.reserve(9);
			
			for (sint32 dy = -1; dy <= 1; dy++) {
				for (sint32 dx = -1; dx <= 1; dx++) {
					const float32 val = temp_disp[(y + dy) * width + x + dx];
					if (val != Invalid_Float) {
						neighbors.push_back(val);
					}
				}
			}
			
			if (!neighbors.empty()) {
				std::nth_element(neighbors.begin(), neighbors.begin() + neighbors.size() / 2, neighbors.end());
				disp_left_[y * width + x] = neighbors[neighbors.size() / 2];
			}
		}
	}
}

void MultiStepRefiner::SimpleMedianFilter()
{
	// ADCE模式下的简化中值滤波
	MedianFilter();
}

const float32* MultiStepRefiner::GetDisparityLeft() const
{
	return disp_left_;
}

const float32* MultiStepRefiner::GetDisparityRight() const
{
	return disp_right_;
}