/* -*-c++-*- AD-Census - Copyright (C) 2020.
* Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
* https://github.com/ethan-li-coding/AD-Census
* Describe	: header of adcensus_util
*/

#pragma once
#include <algorithm>
#include <vector>
#include "adcensus_types.h"


namespace adcensus_util
{
	/**
	* \brief census�任
	* \param source	���룬Ӱ������
	* \param census	�����censusֵ���飬Ԥ����ռ�
	* \param width	���룬Ӱ���
	* \param height	���룬Ӱ���
	*/
	void census_transform_9x7(const uint8* source, std::vector<uint64>& census, const sint32& width, const sint32& height);
	// Hamming����
	uint8 Hamming64(const uint64& x, const uint64& y);

	/**
	* \brief ��ֵ�˲�
	* \param in				���룬Դ����
	* \param out			�����Ŀ������
	* \param width			���룬����
	* \param height			���룬�߶�
	* \param wnd_size		���룬���ڿ���
	*/
	void MedianFilter(const float32* in, float32* out, const sint32& width, const sint32& height, const sint32 wnd_size);
}