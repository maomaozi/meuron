#include "stdafx.h"

#include "ops.h"

namespace ops 
{

	// 构图时推断属性
	template <typename DataType>
	void op::sum_deduction_property(const opNode<DataType> *lhs, const opNode<DataType> *rhs, opNode<DataType> *self)
	{

		/// 无论能否创建节点都先为其命名
		std::string newName;
		newName.append("(");
		newName.append(lhs->name);
		newName.append(" + ");
		newName.append(rhs->name);
		newName.append(")");

		self->setName(std::move(newName));

		/// 检测初始化
		if (!lhs->is_initialized || !rhs->is_initialized) 
		{
			printf("Fatal: Can't create node %s from %s, broken node, Aborting...\n", 
				self->name.c_str(),
				lhs->is_initialized ? rhs->name.c_str() : lhs->name.c_str()
			);

			return;
		}

		/// 检查shape
		if (!checkShapeSame(lhs, rhs))	return;

		self->shape = lhs->shape;

		size_t dataSize = shape2size(self->shape);
		if (dataSize == -1) 
		{
			printf("Error: Broken node %s, not support uncertain shape here\n", self->name.c_str());
			return;
		}

		/// alloc
		try
		{
			if (self->data == nullptr) 
			{
				self->data = std::shared_ptr<DataType>(new DataType[dataSize]);
			}
		}
		catch (const std::exception&)
		{
			printf("Error: OOM when allocate %s, size %.2fMB\n", self->name.c_str(), dataSize / 1024.0 / 1024.0);
			return;
		}

		self->dataSize = dataSize;
		self->is_initialized = true;
	}

	// 实际计算函数
	template <typename DataType>
	void op::sum(const opNode<DataType> *lhs, const opNode<DataType> *rhs, opNode<DataType> *self)
	{
		/// 计算两个矩阵element-wise相加

		/// 检查前置节点是否有值
		if (!lhs->has_data || !rhs->has_data) 
		{
			printf("Fatal: Node %s has no data, Aborting...\n", lhs->has_data ? rhs->name.c_str() : lhs->name.c_str());
			return;
		}

		if (!lhs->is_initialized || !rhs->is_initialized) 
		{
			printf("Fatal: Node %s is broken, Aborting...\n", lhs->is_initialized ? rhs->name.c_str() : lhs->name.c_str());
			return;
		}

		/// 实际计算
		DataType * res = self->data.get();
		DataType * lv = lhs->data.get();
		DataType * rv = rhs->data.get();

		for (size_t i = 0; i < self->dataSize; ++i) 
		{
			res[i] = lv[i] + rv[i];
		}

		/// 此节点本次推理已经计算值
		self->has_data = true;
	}

	// API
	template<typename T>
	Var<T> op::matSum(Tensor &lhs, Tensor &rhs)
	{
		// 首先产生正确绑定的新节点
		__Node *lhsNode = lhs.getNode();
		opNode<T> *lhsOpNode = (opNode<T> *)(lhsNode);

		__Node *rhsNode = rhs.getNode();
		opNode<T> *rhsOpNode = (opNode<T> *)(rhsNode);

		opNode<T> &res = lhsOpNode->bind(*rhsOpNode, op::sum);

		// 然后推断新节点的属性
		// 包括shape name, 然后认为此节点已经初始化完毕
		op::sum_deduction_property(lhsOpNode, rhsOpNode, &res);

		return Var<T>(res);
	}

}


template<typename T>
opNode<T> &opNode<T>::operator+(opNode<T> &rhsObj)
{
	// 首先产生正确绑定的新节点
	opNode<T> &res = bind(rhsObj, ops::op::sum);

	// 然后推断新节点的属性
	// 包括shape name, 然后认为此节点已经初始化完毕
	ops::op::sum_deduction_property(this, &rhsObj, &res);

	return res;
}


template<typename DataType>
inline Var<DataType> Var<DataType>::operator+(Var<DataType>& rhsObj)
{
	opNode<DataType> &res = *_node + *rhsObj._node;

	return Var<DataType>(res);
	//return res;
}