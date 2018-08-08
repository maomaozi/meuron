#pragma once

#include "stdafx.h"

template <typename T>
class opNode;

namespace ops {

	class op
	{
	public:

		op() {};
		~op() {};

	public:
		template <typename T>
		static void sum(const opNode<T> *lhs, const opNode<T> *rhs, opNode<T> *self);

		template <typename T>
		static void sum_deduction_property(const opNode<T> *lhs, const opNode<T> *rhs, opNode<T> *self);


	};
}