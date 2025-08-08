/* -*-c++-*- AD-Census - Copyright (C) 2020.
* Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
* https://github.com/ethan-li-coding/AD-Census
* Describe	: header of class CrossAggregator
*/

#ifndef AD_CENSUS_CROSS_AGGREGATOR_H_
#define AD_CENSUS_CROSS_AGGREGATOR_H_

#include "adcensus_types.h"
#include <algorithm>
#include <vector>

/**
* \brief ����ʮ�ֱ۽ṹ
* Ϊ�������ڴ�ռ�ã��۳���������Ϊuint8������ζ�ű۳�����ܳ���255
*/

/**\brief ���۳� */
#define MAX_ARM_LENGTH 255 

/**
 * \brief ʮ�ֽ�������۾ۺ���
 */
class CrossAggregator {
public:
	CrossAggregator();
	~CrossAggregator();

	/**
	 * \brief ��ʼ�����۾ۺ���
	 * \param width		Ӱ���
	 * \param height	Ӱ���
	 * \return true:��ʼ���ɹ�
	 */
	bool Initialize(const sint32& width, const sint32& height, const sint32& min_disparity, const sint32& max_disparity);

	/**
	 * \brief ���ô��۾ۺ���������
	 * \param img_left		// ��Ӱ�����ݣ���ͨ��
	 * \param img_right		// ��Ӱ�����ݣ���ͨ��
	 * \param cost_init		// ��ʼ��������
	 */
	void SetData(const uint8* img_left, const uint8* img_right, const float32* cost_init);
	
	/**
	 * \brief 设置代价聚合器数据（重载版本）
	 * \param cost_left		// 左影像代价数据
	 * \param cost_right	// 右影像代价数据
	 */
	void SetData(const float32* cost_left, const float32* cost_right);

	/**
	 * \brief ���ô��۾ۺ����Ĳ���
	 * \param cross_L1		// L1
	 * \param cross_L2		// L2
	 * \param cross_t1		// t1
	 * \param cross_t2		// t2
	 */
	void SetParams(const sint32& cross_L1, const sint32& cross_L2, const sint32& cross_t1, const sint32& cross_t2);

	/** \brief �ۺ� */
	void Aggregate(const sint32& num_iters);

	/** \brief ��ȡ�������ص�ʮ�ֽ��������ָ�� */
	CrossArm* get_arms_ptr();

	/** \brief ��ȡ�ۺϴ�������ָ�� */
	float32* get_cost_ptr();
	
	/** \brief 获取左影像聚合代价数据 */
	const float32* GetCostLeft() const;
	
	/** \brief 获取右影像聚合代价数据 */
	const float32* GetCostRight() const;
private:
	/** \brief ����ʮ�ֽ���� */
	void BuildArms();
	/** \brief ����ˮƽ�� */
	void FindHorizontalArm(const sint32& x, const sint32& y, uint8& left, uint8& right) const;
	/** \brief ������ֱ�� */
	void FindVerticalArm(const sint32& x, const sint32& y, uint8& top, uint8& bottom) const;
	/** \brief �������ص�֧������������ */
	void ComputeSupPixelCount();
	/** \brief �ۺ�ĳ���Ӳ� */
	void AggregateInArms(const sint32& disparity, const bool& horizontal_first);

	/** \brief ������ɫ���� */
	inline sint32 ColorDist(const ADColor& c1,const ADColor& c2) const {
		return std::max(abs(c1.r - c2.r), std::max(abs(c1.g - c2.g), abs(c1.b - c2.b)));
	}
private:
	/** \brief ͼ��ߴ� */
	sint32	width_;
	sint32	height_;

	/** \brief ����� */
	std::vector<CrossArm> vec_cross_arms_;

	/** \brief Ӱ������ */
	const uint8* img_left_;
	const uint8* img_right_;

	/** \brief ��ʼ��������ָ�� */
	const float32* cost_init_;
	/** \brief �ۺϴ������� */
	std::vector<float32> cost_aggr_;

	/** \brief ��ʱ�������� */
	std::vector<float32> vec_cost_tmp_[2];
	/** \brief ֧���������������� 0��ˮƽ������ 1����ֱ������ */
	std::vector<uint16> vec_sup_count_[2];
	std::vector<uint16> vec_sup_count_tmp_;

	sint32	cross_L1_;			// ʮ�ֽ��洰�ڵĿռ��������L1
	sint32  cross_L2_;			// ʮ�ֽ��洰�ڵĿռ��������L2
	sint32	cross_t1_;			// ʮ�ֽ��洰�ڵ���ɫ�������t1
	sint32  cross_t2_;			// ʮ�ֽ��洰�ڵ���ɫ�������t2
	sint32  min_disparity_;			// ��С�Ӳ�
	sint32	max_disparity_;			// ����Ӳ�

	/** \brief �Ƿ�ɹ���ʼ����־	*/
	bool is_initialized_;
};
#endif
