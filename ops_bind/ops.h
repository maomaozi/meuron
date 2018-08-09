#pragma once

#include "stdafx.h"
#include "utils.h"
#include "types.h"

#define ABORT_IF_FAILURE(x) if(!(x)) return;

template <typename T>
class opNode;

namespace ops 
{

	class op
	{
	public:

		op() {};
		~op() {};

	public:

		static bool check_initialized(const __Node * lhs, const __Node * rhs, const std::string &self_name)
		{
			if (!lhs->get_initialized() || !rhs->get_initialized())
			{
				printf("Fatal: Can't create node %s from %s, broken node, Aborting...\n",
					self_name.c_str(),
					lhs->get_initialized() ? rhs->get_name().c_str() : lhs->get_name().c_str()
				);

				return false;
			}

			return true;
		}


		static bool check_initialized(const __Node * lhs, const std::string &self_name)
		{
			if (!lhs->get_initialized())
			{
				printf("Fatal: Can't create node %s from %s, broken node, Aborting...\n",
					self_name.c_str(), lhs->get_name().c_str()
				);

				return false;
			}

			return true;
		}


		static bool check_shape(const __Node *lhs, const __Node *rhs, std::vector<int> expect_shape)
		{
			/// 形状相同非空并且符合预期

			bool flag = true;

			/// 判断矩阵是否为空
			if (!lhs->get_shape().size())
			{
				flag = false;
			}

			/// 形状相同
			if (lhs->get_shape() != expect_shape)
			{
				flag = false;
			}

			/// 打印错误信息
			if (!flag)
			{
				printf("Fatal: %s's dim is not compactible with %s: %s vs %s, Aborting...\n",
					lhs->get_name().c_str(),
					rhs->get_name().c_str(),
					vec_shape_to_str(lhs->get_shape()).c_str(),
					vec_shape_to_str(rhs->get_shape()).c_str()
				);
			}

			return flag;
		}


		static bool check_shape(const __Node *lhs, std::vector<int> expect_shape)
		{
			/// 形状相同非空并且符合预期

			bool flag = true;

			/// 判断矩阵是否为空
			if (!lhs->get_shape().size())
			{
				flag = false;
			}

			/// 形状相同
			if (lhs->get_shape() != expect_shape)
			{
				flag = false;
			}

			/// 打印错误信息
			if (!flag)
			{
				printf("Fatal: %s's dim is not compactible: %s, Aborting...\n",
					lhs->get_name().c_str(),
					vec_shape_to_str(lhs->get_shape()).c_str()
				);
			}

			return flag;
		}


		static bool check_has_data(const __Node *lhs, const std::string &self_name)
		{
			if (!lhs->get_calculated())
			{
				printf("Fatal: Parent node %s of %s has no data, Aborting...\n", lhs->get_name().c_str(), self_name.c_str());
				return false;
			}

			return true;
		}

	};
}