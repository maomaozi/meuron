#pragma once

#include "stdafx.h"

typedef		__int8		INT8_t;
typedef		__int32		INT32_t;
typedef		float		FLOAT32_t;
typedef		double		FLOAT64_t;


enum dataType
{
	UNKOWN = 0,
	INT8,
	INT32,
	FLOAT32,
	FLOAT64
};

/*
template<typename T>
dataType get_type() {
	return UNKOWN;
}

template<>
dataType get_type<INT8_t>() {
	return INT8;
}

template<>
dataType get_type<INT32_t>() {
	return INT32;
}

template<>
dataType get_type<FLOAT32_t>() {
	return FLOAT32;
}

template<>
dataType get_type<FLOAT64_t>() {
	return FLOAT64;
}
*/

//=============================================
