#include "stdafx.h"

#include "tensor.h"


template<typename DataType>
inline Var<DataType>::Var(opNode<DataType>& node)
{
	_node = &node;
}

template<typename DataType>
inline Var<DataType>::Var(Var<DataType>& rhsVar)
{
	_node = rhsVar._node;
	_saver = rhsVar._saver;
}

template<typename DataType>
inline Var<DataType>::Var(Var<DataType>&& rhsVar)
{
	_node = rhsVar._node;
	_saver = std::move(rhsVar._saver);
}

template<typename DataType>
inline Var<DataType>::Var(const std::vector<int>& initShape, std::string name)
{
	_saver = std::shared_ptr<opNode<DataType>>(new opNode<DataType>(nullptr, initShape, name));
	_node = _saver.get();

	size_t data_size = _node->get_datasize();

	if (!data_size) 
	{
		printf("Error: Can't allocate memory for %s, error shape\n", name.c_str());
		return;
	}

	///  新创建节点时为其分配内存

#ifdef USE_CUDA

	DataType* tmp = nullptr;
	size_t bytes = data_size * sizeof(DataType);
	cudaMalloc((void **)&tmp, bytes);
	_node->data = std::shared_ptr<DataType>(tmp, [](void *p) {cudaFree(p); });
#else

	_node->data = std::shared_ptr<DataType>(new DataType[data_size], [](void *p) { delete[] p; });

#endif // DEBUG

	_node->is_calculated = true;
}


//===================================


template<typename DataType>
void Var<DataType>::init(DataType *data, bool is_ref)
{
	if (!_node->is_initialized) 
	{
		printf("Error: Var %s is broken\n", _node->name.c_str());

		return;
	}

#ifdef USE_CUDA

	if (is_ref)
	{
		printf("Error: Var %s can not malloc memory is it's a ref\n", _node->name.c_str());
		return;
	}

	CHECK(cudaMemcpy(_node->data.get(), data, _node->data_size * sizeof(DataType), cudaMemcpyHostToDevice));
	
#else

	if (is_ref)
	{
		/// 如果使用外部数据的引用
		_node->data = data;
		_node->is_calculated = true;

		return;
	}

	DataType *mem = _node->data.get();

	for (size_t i = 0; i < _node->get_datasize(); ++i) 
	{
		mem[i] = data[i];
	}

#endif // USE_CUDA

	_node->is_calculated = true;
}


/*
template<typename DataType>
void Var<DataType>::init(const std::initializer_list<DataType> &data)
{
	if (!_node->is_initialized) {
		printf("Error: Var %s is broken\n", _node->name.c_str());

		return;
	}

#ifdef USE_CUDA

	const DataType *tmp = &data[0];

	CHECK(cudaMemcpy(_node->data.get(), tmp, _node->get_datasize() * sizeof(DataType), cudaMemcpyHostToDevice));

#else

	DataType *mem = _node->data.get();

	int i = 0, max_i = _node->get_datasize();

	for (const auto &iter : data) 
	{
		if (i > max_i) break;
		mem[i] = iter;
		++i;
	}

#endif

	_node->is_calculated = true;
}
*/

template<typename DataType>
void Var<DataType>::init(const std::vector<DataType> &data)
{
	if (!_node->is_initialized) {
		printf("Error: Var %s is broken\n", _node->name.c_str());

		return;
	}


#ifdef USE_CUDA
	const DataType *tmp = &data[0];

	CHECK(cudaMemcpy(_node->data.get(), tmp, _node->get_datasize() * sizeof(DataType), cudaMemcpyHostToDevice));

#else
	DataType *mem = _node->data.get();

	for (size_t i = 0; i < _node->get_datasize(); ++i)
	{
		mem[i] = data[i];
	}
#endif

	_node->is_calculated = true;
}

//====================================================


template<typename DataType>
inline opNode<DataType> * Var<DataType>::getNode() {
	return _node;
}

//====================================================