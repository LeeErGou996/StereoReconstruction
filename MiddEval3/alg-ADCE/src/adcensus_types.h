/* -*-c++-*- AD-Census - Copyright (C) 2020.
* Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
* https://github.com/ethan-li-coding/AD-Census
* Describe	: AD-Census立体匹配算法的基础类型定义
*/

#ifndef ADCENSUS_TYPES_H_
#define ADCENSUS_TYPES_H_

#include <cstdint>
#include <limits>
#include <algorithm>
#include <cmath>



//··············································································//
// 基础数据类型定义
typedef int8_t			sint8;		// 有符号8位整数
typedef uint8_t			uint8;		// 无符号8位整数
typedef int16_t			sint16;		// 有符号16位整数
typedef uint16_t		uint16;		// 无符号16位整数
typedef int32_t			sint32;		// 有符号32位整数
typedef uint32_t		uint32;		// 无符号32位整数
typedef int64_t			sint64;		// 有符号64位整数
typedef uint64_t		uint64;		// 无符号64位整数
typedef float			float32;	// 32位浮点数
typedef double			float64;	// 64位浮点数

//··············································································//
// 常量定义
constexpr auto Invalid_Float = std::numeric_limits<float32>::infinity();
constexpr auto Large_Float = 99999.0f;

//··············································································//
// 安全删除宏
#define SAFE_DELETE(P) {if(P) delete[](P);(P)=nullptr;}

//··············································································//
/**
 * \brief AD-Census算法配置参数
 */
struct ADCensusOption {
	sint32	min_disparity;		// 最小视差
	sint32	max_disparity;		// 最大视差

	// 代价计算参数
	float32 lambda_ad;			// AD代价权重
	float32 lambda_census;		// Census代价权重

	// 十字交叉窗口聚合参数
	sint32	cross_L1;			// 十字交叉窗口臂长参数L1
	sint32	cross_L2;			// 十字交叉窗口臂长参数L2
	sint32	cross_t1;			// 十字交叉窗口颜色阈值t1
	sint32	cross_t2;			// 十字交叉窗口颜色阈值t2

	// 扫描线优化参数
	float32 so_p1;				// 扫描线优化惩罚项P1
	float32 so_p2;				// 扫描线优化惩罚项P2  
	sint32  so_tso;				// 扫描线优化纹理阈值

	// 迭代区域投票法参数
	sint32	irv_ts;				// 迭代区域投票法纹理阈值
	float32	irv_th;				// 迭代区域投票法视差阈值

	// 一致性检验参数
	float32	lrcheck_thres;		// 左右一致性检验阈值

	// 处理选项
	bool	do_lr_check;		// 是否执行左右一致性检验
	bool	do_filling;			// 是否执行视差填充
	bool	do_discontinuity_adjustment;	// 是否执行非连续区域调整

	/** \brief 默认参数构造函数 */
	ADCensusOption(): min_disparity(0), max_disparity(64), 
					  lambda_ad(10.0f), lambda_census(30.0f),
					  cross_L1(34), cross_L2(17), cross_t1(20), cross_t2(6),
					  so_p1(1.0f), so_p2(3.0f), so_tso(15),
					  irv_ts(20), irv_th(0.4f),
					  lrcheck_thres(1.0f),
					  do_lr_check(true), do_filling(true), do_discontinuity_adjustment(false) { }
};

//··············································································//
/**
 * \brief 颜色结构体
 */
struct ADColor {
	uint8 r, g, b;
	
	ADColor() : r(0), g(0), b(0) { }
	ADColor(uint8 _r, uint8 _g, uint8 _b) : r(_r), g(_g), b(_b) { }
	
	// 相等比较
	bool operator==(const ADColor& other) const {
		return r == other.r && g == other.g && b == other.b;
	}
	
	// 不等比较
	bool operator!=(const ADColor& other) const {
		return !(*this == other);
	}
};

//··············································································//
/**
 * \brief 交叉臂结构体
 */
struct CrossArm {
	uint8 left;		// 左臂长度
	uint8 right;	// 右臂长度
	uint8 top;		// 上臂长度
	uint8 bottom;	// 下臂长度
	
	CrossArm() : left(0), right(0), top(0), bottom(0) { }
	CrossArm(uint8 l, uint8 r, uint8 t, uint8 b) : left(l), right(r), top(t), bottom(b) { }
};

//··············································································//
/**
 * \brief 像素结构体
 */
struct ADPixel {
	sint32 x, y;
	
	ADPixel() : x(0), y(0) { }
	ADPixel(sint32 _x, sint32 _y) : x(_x), y(_y) { }
};

//··············································································//
/**
 * \brief 计算两个颜色之间的距离
 * \param c1 颜色1
 * \param c2 颜色2  
 * \return 颜色距离
 */
inline uint8 ColorDist(const ADColor& c1, const ADColor& c2) {
	const int dr = static_cast<int>(c1.r) - static_cast<int>(c2.r);
	const int dg = static_cast<int>(c1.g) - static_cast<int>(c2.g);
	const int db = static_cast<int>(c1.b) - static_cast<int>(c2.b);
	return static_cast<uint8>(std::max(std::max(abs(dr), abs(dg)), abs(db)));
}

#endif // ADCENSUS_TYPES_H_