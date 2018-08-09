#include "stdafx.h"
#include "utils.h"




std::string vecShape2Str(const std::vector<int>& vec) 
{
	if (vec.empty()) 
	{
		return "(NULL)";
	}

	std::string res = "(";
	char tmp[10];
	for (size_t i = 0; i < vec.size(); ++i) 
	{
		itoa(vec[i], tmp, 10);
		res.append(tmp);
		if (i != vec.size() - 1) 
		{
			res.append(", ");
		}
	}
	res.append(")");

	return res;
}


template<typename U, typename V>
bool checkTypeSame() 
{
	return std::is_same<U, V>::value;
}

bool checkShapeSame(const __Node *lhs, const __Node *rhs)
{
	/// 检查两个矩阵是否形状相同并且非空

	bool flag = true;

	/// 判断矩阵是否为空
	if (!lhs->getShape().size() || !rhs->getShape().size())
	{
		flag = false;
	}

	/// 两边操作数形状相同
	if (lhs->getShape() != rhs->getShape())
	{
		flag = false;
	}

	/// 打印错误信息
	if (!flag)
	{
		printf("Fatal: %s's dim is not compactible with %s: %s vs %s, Aborting...\n",
			lhs->getName().c_str(),
			rhs->getName().c_str(),
			vecShape2Str(lhs->getShape()).c_str(),
			vecShape2Str(rhs->getShape()).c_str()
		);
	}

	return flag;
}

/*
bool checkInitialized(const __Node * lhs, const __Node * rhs)
{
	if (!lhs->get_initialized() || !rhs->get_initialized())
	{
		printf("Fatal: Can't create node %s from %s, broken node, Aborting...\n",
			self->name.c_str(),
			lhs->get_initialized() ? rhs->name.c_str() : lhs->name.c_str()
		);

		return;
	}
}
*/

size_t shape2size(const std::vector<int>& shape) 
{
	size_t dataSizeAccu = 1;

	for (const int &i : shape) 
	{
		if (i <= 0)
		{
			return -1;
		}
		dataSizeAccu *= i;
	}

	return dataSizeAccu;
}