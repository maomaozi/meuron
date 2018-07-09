#include "stdafx.h"

#include "ops.h"

namespace ops {

	template <typename T>
	void op::sum_deduction_property(const opNode<T> *lhs, const opNode<T> *rhs, opNode<T> *self) {

		/// 无论能否创建节点都先为其命名
		std::string newName;
		newName.append("(");
		newName.append(lhs->name);
		newName.append(" + ");
		newName.append(rhs->name);
		newName.append(")");

		self->setName(std::move(newName));

		if (!lhs->is_initialized || !rhs->is_initialized) {
			printf("Fatal: Can't create node %s from %s, broken node, Aborting...\n", 
				self->name.c_str(),
				lhs->is_initialized ? rhs->name.c_str() : lhs->name.c_str()
			);

			return;
		}

		/// 检查shape
		if (!checkShapeSame(lhs->shape, rhs->shape)) {
			printf("Fatal: %s's dim is not compactible with %s: %s vs %s, Aborting...\n",
				lhs->getName().c_str(),
				rhs->getName().c_str(),
				vecShape2Str(lhs->shape).c_str(),
				vecShape2Str(rhs->shape).c_str()
			);
			return;
		}

		self->shape = lhs->shape;

		size_t dataSize = shape2size(self->shape);
		if (dataSize == -1) {
			printf("Error: Broken node %s, not support uncertain shape here\n", self->name.c_str());
			return;
		}

		/// alloc
		try
		{
			if (self->data == nullptr) {
				self->data = std::shared_ptr<T>(new T[dataSize]);
			}
		}
		catch (const std::exception&)
		{
			printf("Error: OOM when allocate %s, size %.2fMB\n", self->name.c_str(), dataSize/1024.0/1024.0);
			return;
		}

		self->dataSize = dataSize;
		self->is_initialized = true;
	}

	template <typename T>
	void op::sum(const opNode<T> *lhs, const opNode<T> *rhs, opNode<T> *self)
	{
		/// 计算两个矩阵element-wise相加

		/// 检查前置节点是否有值
		if (!lhs->has_data || !rhs->has_data) {
			printf("Fatal: Node %s has no data, Aborting...\n", lhs->has_data ? rhs->name.c_str() : lhs->name.c_str());
			return;
		}

		if (!lhs->is_initialized || !rhs->is_initialized) {
			printf("Fatal: Node %s is broken, Aborting...\n", lhs->is_initialized ? rhs->name.c_str() : lhs->name.c_str());
			return;
		}

		/// 实际计算
		T * res = self->data.get();
		T * lv = lhs->data.get();
		T * rv = rhs->data.get();

		for (size_t i = 0; i < self->dataSize; ++i) {
			res[i] = lv[i] + rv[i];
		}

		/// 此节点本次推理已经计算值
		self->has_data = true;
	}
}