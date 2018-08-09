#include "stdafx.h"
#include "ops.h"

#ifdef USE_CUDA
#include "mat_sum.h"
#endif

namespace ops 
{

	class sum_op final : public op 
	{
	public:
		template <typename T>
		static void inferer(const opNode<T> *lhs, const opNode<T> *rhs, opNode<T> *self)
		{
			/// 无论能否创建节点都先为其命名
			std::string newName;
			newName.append("(");
			newName.append(lhs->get_name());
			newName.append(" + ");
			newName.append(rhs->get_name());
			newName.append(")");

			self->set_name(std::move(newName));

			/// 检测初始化
			ABORT_IF_FAILURE(check_initialized(lhs, lhs->get_name()));
			ABORT_IF_FAILURE(check_initialized(rhs, rhs->get_name()));

			/// 检查shape
			ABORT_IF_FAILURE(check_shape(lhs, rhs->get_shape()));

			self->set_shape(lhs->get_shape());

			size_t data_size = shape_to_size(self->get_shape());

			if (data_size == -1)
			{
				printf("Error: Broken node %s, not support uncertain shape here\n", self->get_name().c_str());
				return;
			}

#ifdef USE_CUDA
			//如果检测到CUDA，分配GPU内存
			size_t bytes = data_size * sizeof(T);

			T *gpu_result = nullptr;

			CHECK(cudaMalloc((T **)&gpu_result, bytes));

			self->get_data() = std::shared_ptr<T>(gpu_result, [](T *ptr) { cudaFree(ptr); });

#else
			/// alloc
			try
			{
				if (self->data == nullptr)
				{
					self->data = std::shared_ptr<T>(new T[data_size]);
				}
			}
			catch (const std::exception&)
			{
				printf("Error: OOM when allocate %s, size %.2fMB\n", self->name.c_str(), data_size / 1024.0 / 1024.0);
				return;
			}
#endif // USE_CUDA

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

			T *res	=	self->get_data().get();
			T *lv	=	lhs->fetch_data().get();
			T *rv	=	rhs->fetch_data().get();

#ifdef USE_CUDA

			gpu_mat_sum(lv, rv, res, self->get_datasize());

#else
			/// 实际计算
			size_t len = self->get_datasize();

			for (size_t i = 0; i < len; ++i)
			{
				res[i] = lv[i] + rv[i];
			}
#endif

			/// 此节点本次推理已经计算值
			self->set_calculated(true);
		}


		template<typename T>
		static Var<T> mat_sum(Tensor &lhs, Tensor &rhs)
		{
			// 首先产生正确绑定的新节点
			__Node *lhsNode = lhs.getNode();
			opNode<T> *lhsOpNode = (opNode<T> *)(lhsNode);

			__Node *rhsNode = rhs.getNode();
			opNode<T> *rhsOpNode = (opNode<T> *)(rhsNode);

			opNode<T> &res = lhsOpNode->bind(*rhsOpNode, sum_op::forward);

			// 然后推断新节点的属性
			// 包括shape name, 然后认为此节点已经初始化完毕
			sum_op::inferer(lhsOpNode, rhsOpNode, &res);

			return Var<T>(res);
		}
	};

}

//=========================================================================


/*
	如果需要重载基本运算符，这里写定义
*/

template<typename T>
opNode<T> &opNode<T>::operator+(opNode<T> &rhsObj)
{
	// 首先产生正确绑定的新节点
	opNode<T> &res = bind(rhsObj, ops::sum_op::forward);

	// 然后推断新节点的属性
	// 包括shape name, 然后认为此节点已经初始化完毕
	ops::sum_op::inferer(this, &rhsObj, &res);

	return res;
}


template<typename T>
inline Var<T> Var<T>::operator+(Var<T>& rhsObj)
{
	opNode<T> &res = *_node + *rhsObj._node;

	return Var<T>(res);
}