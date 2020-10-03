#include "stdafx.h"
#include "opnode.h"

//===========================================================
	template<typename T>
	opNode<T>::opNode()
	{
		;
	}

	template<typename T>
	opNode<T>::opNode(const opNode<T> &rhsNode)
	{
		lhs = rhsNode.lhs;
		rhs = rhsNode.rhs;

		data = rhsNode.data;
		data_size = rhsNode.data_size;

		next = rhsNode.next;

		func_mono = rhsNode.func_mono;
		func_bino = rhsNode.func_bino;

		shape = rhsNode.shape;
		name = rhsNode.name;
		is_calculated = rhsNode.is_calculated;
		is_initialized = rhsNode.is_initialized;
	}

	template<typename T>
	opNode<T>::opNode(T *initData, const std::vector<int> &initShape, const std::string &initName) : data(initData), name(initName), shape(initShape)
	{
		size_t dataSizeAccu = 1;

		for (const int &i : shape) {
			if (i <= 0)
			{
				printf("Error: Broken node %s, not support uncertain shape here\n", name.c_str());
				return;
			}
			dataSizeAccu *= i;
		}

		data_size = dataSizeAccu;

		if (data) {
			is_calculated = true;
		}

		is_initialized = true;
	}

	template<typename T>
	opNode<T>::opNode(opNode<T> * const lhsObj, operator_mono_t op) :
		lhs(lhsObj), func_mono(op)
	{
		;
	}

	template<typename T>
	opNode<T>::opNode(opNode<T> * const lhsObj, opNode<T> * const rhsObj, operator_bino_t op) :
		lhs(lhsObj), rhs(rhsObj), func_bino(op)
	{
		;
	}

	template<typename T>
	opNode<T>::~opNode()
	{

	}

//===========================================================

	template<typename T>
	opNode<T> &opNode<T>::bind(opNode<T> &rhsObj, operator_bino_t op)
	{
		// 双目bind, 在两个前驱节点上保存后继节点
		// 如果不保存这个shared_ptr，当函数返回后引用计数为0
		// 对于流图内部绑定产生的节点，在完成最终属性推断后认为其已经初始化

		//next = std::shared_ptr<opNode<T>>(new opNode<T>(this, &rhsObj, op));
		//rhsObj.next = next;

		auto newNode = std::shared_ptr<opNode<T>>(new opNode<T>(this, &rhsObj, op));

		next.insert(newNode);
		rhsObj.next.insert(newNode);

		return *newNode;
	}

	template<typename T>
	opNode<T> &opNode<T>::bind(operator_mono_t op)
	{
		// 单目bind
		// 处理类似于双目运算
		// 对于流图内部绑定产生的节点，在完成最终属性推断后认为其已经初始化
		//next = std::shared_ptr<opNode<T>>(new opNode<T>(this, op));
		auto newNode = std::shared_ptr<opNode<T>>(new opNode<T>(this, op));
		next.insert(newNode);

		return *newNode;
	}

	template<typename T>
	void opNode<T>::exec()
	{
		if (!lhs) return;

		if (!rhs) {
			lhs->exec();
			func_mono(lhs, this);
		}

		lhs->exec();
		rhs->exec();
		func_bino(lhs, rhs, this);
	}

	template<typename T>
	void opNode<T>::exec_here()
	{
		if (!lhs) return;

		if (!rhs) {
			func_mono(lhs, this);
		}

		func_bino(lhs, rhs, this);
	}

	template<typename T>
	void opNode<T>::repr()
	{
		if (is_initialized) 
		{
			printf("Node %s\t%s\n", name.c_str(), vec_shape_to_str(shape).c_str());
			if (is_calculated) 
			{
				printf("values: ");
#ifdef USE_CUDA
				T *result = new T[data_size];
				CHECK(cudaMemcpy(result, data.get(), sizeof(T) * data_size, cudaMemcpyDeviceToHost));
				
				for (size_t i = 0; i < data_size; i++)
				{
					std::cout << (int)result[i] << std::endl;
				}

				delete[]result;
#else
				for (size_t i = 0; i < data_size; ++i) 
				{
					std::cout << (int)data.get()[i] << std::endl;
				}
#endif
			}
		}
		
	}

	template<typename T>
	T & opNode<T>::operator=(opNode<T> rhsObj)
	{
		swap(rhsObj);

		return *this;
	}

	template<typename T>
	bool opNode<T>::operator==(const opNode<T> & rhsNode)
	{
		// TODO: 判定形状
		return &data == &(rhsNode.data);
	}

	template<typename T>
	void opNode<T>::operator()()
	{
		exec();
	}

	template<typename T>
	void opNode<T>::swap(opNode<T> &lhsNode, opNode<T> &rhsNode)
	{
		using std::swap;

		swap(lhsNode.lhs, rhsNode.lhs);
		swap(lhsNode.rhs, rhsNode.rhs);

		swap(lhsNode.data, rhsNode.data);
		swap(lhsNode.data_size, rhsNode.data_size);

		swap(lhsNode.next, rhsNode.next);

		swap(lhsNode.func_mono, rhsNode.func_mono);
		swap(lhsNode.func_bino, rhsNode.func_bino);

		swap(lhsNode.name, rhsNode.name);
		swap(lhsNode.shape, rhsNode.shape);
		swap(lhsNode.is_calculated, rhsNode.is_calculated);
		swap(lhsNode.is_initialized, rhsNode.is_initialized);


	}

	template<typename T>
	const std::string &opNode<T>::get_name() const
	{
		return name;
	}

	template<typename T>
	const std::vector<int> &opNode<T>::get_shape() const
	{
		return shape;
	}

	template<typename T>
	bool opNode<T>::get_initialized() const
	{
		return is_initialized;
	}

	template<typename T>
	std::shared_ptr<T> &opNode<T>::get_data()
	{
		return data;
	}

	template<typename T>
	const std::shared_ptr<T> &opNode<T>::fetch_data() const
	{
		return data;
	}


	template<typename T>
	size_t opNode<T>::get_datasize() const 
	{
		return data_size;
	}

	template<typename T>
	bool opNode<T>::get_calculated() const
	{
		return is_calculated;
	}

	template<typename T>
	const __Node* opNode<T>::get_lhs_parent() const
	{
		return lhs;
	}

	template<typename T>
	const __Node* opNode<T>::fetch_rhs_parent() const
	{
		return rhs;
	}

	template<typename T>
	const std::unordered_set<std::shared_ptr<__Node>> &opNode<T>::fetchNext() const
	{
		return next;
	}

	template<typename T>
	__Node *opNode<T>::get_lhs_parent() 
	{
		return lhs;
	}

	template<typename T>
	__Node *opNode<T>::fetch_rhs_parent()  
	{
		return rhs;
	}

	template<typename T>
	std::unordered_set<std::shared_ptr<__Node>> &opNode<T>::fetchNext() 
	{
		return next;
	}

	template<typename T>
	void opNode<T>::set_name(const std::string &name) 
	{
		this->name = name;
	}

	template<typename T>
	void opNode<T>::set_shape(const std::vector<int>& shape)
	{
		this->shape = shape;
	}

	template<typename T>
	void opNode<T>::set_datasize(size_t data_size)
	{
		this->data_size = data_size;
	}

	template<typename T>
	void opNode<T>::set_initialized(bool is_initialized)
	{
		this->is_initialized = is_initialized;
	}

	template<typename T>
	void opNode<T>::set_calculated(bool is_calculated)
	{
		this->is_calculated = is_calculated;
	}


	template<typename T>
	void opNode<T>::reset()
	{
		this->is_calculated = false;
	}
