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
			/// �����ܷ񴴽��ڵ㶼��Ϊ������
			std::string newName;
			newName.append("reshape");
			newName.append(":");
			newName.append(self->get_name());
			self->set_name(std::move(newName));

			/// ����ʼ��
			ABORT_IF_FAILURE(check_initialized(lhs, lhs->get_name()));

			/// �����Լ���
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
			/// ������������element-wise���

			/// ���ǰ�ýڵ��Ƿ���ֵ
			ABORT_IF_FAILURE(check_has_data(lhs, self->get_name()));
			ABORT_IF_FAILURE(check_has_data(rhs, self->get_name()));

			/// �˽ڵ㱾�������Ѿ�����ֵ
			self->set_calculated(true);
		}


		template<typename T>
		static Var<T> reshape(Tensor &lhs, std::vector<int> new_shape)
		{
			// ���Ȳ�����ȷ�󶨵��½ڵ�
			__Node *lhsNode = lhs.getNode();
			opNode<T> *lhsOpNode = (opNode<T> *)(lhsNode);

			opNode<T> &res = lhsOpNode->bind(reshape_op::forward);

			// Ȼ���ƶ��½ڵ������
			// ����shape name, Ȼ����Ϊ�˽ڵ��Ѿ���ʼ�����
			reshape_op::inferer(lhsOpNode, res, new_shape);

			return Var<T>(res);
		}
	};

}
