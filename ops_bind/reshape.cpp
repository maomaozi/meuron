#include "stdafx.h"
#include "ops.h"

namespace ops
{
	class reshape_op final : public op
	{
	public:
		template <typename T>
		static void inferer(const opNode<T> *lhs, opNode<T> *self, std::vector<int> new_shape)
		{
			/// 无论能否创建节点都先为其命名
			std::string newName;
			newName.append("reshape");
			newName.append(":");
			newName.append(self->get_name());
			self->set_name(std::move(newName));

			/// 检测初始化
			ABORT_IF_FAILURE(check_initialized(lhs, lhs->get_name()));

			/// 相容性计算
			ABORT_IF_FAILURE(shape_to_size(lhs->get_shape()) == shape_to_size(new_shape));


			self->set_shape(new_shape);

			size_t data_size = shape_to_size(self->get_shape());

			if (data_size == -1)
			{
				printf("Error: Broken node %s, not support uncertain shape here\n", self->get_name().c_str());
				return;
			}

			self->set_datasize(data_size);
			self->set_initialized(true);
		}


		template <typename T>
		static void forward(const opNode<T> *lhs, const opNode<T> *rhs, opNode<T> *self)
		{
			/// 计算两个矩阵element-wise相加

			/// 检查前置节点是否有值
			ABORT_IF_FAILURE(check_has_data(lhs, self->get_name()));
			ABORT_IF_FAILURE(check_has_data(rhs, self->get_name()));

			/// 此节点本次推理已经计算值
			self->set_calculated(true);
		}


		template<typename T>
		static Var<T> reshape(Tensor &lhs, std::vector<int> new_shape)
		{
			// 首先产生正确绑定的新节点
			__Node *lhsNode = lhs.getNode();
			opNode<T> *lhsOpNode = (opNode<T> *)(lhsNode);

			opNode<T> &res = lhsOpNode->bind(reshape_op::forward);

			// 然后推断新节点的属性
			// 包括shape name, 然后认为此节点已经初始化完毕
			reshape_op::inferer(lhsOpNode, res, new_shape);

			return Var<T>(res);
		}
	};

}
