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

	size_t dataSize = _node->getDataSize();

	if (!dataSize) {
		printf("Error: Can't allocate memory for %s, error shape\n", name.c_str());
		return;
	}

	///  �´����ڵ�ʱΪ������ڴ�
	_node->data = std::shared_ptr<DataType>(new DataType[dataSize], [](void *p) { delete [] p; });

	_shape = initShape;
}


//===================================


template<typename DataType>
void Var<DataType>::init(DataType *data, bool is_ref)
{
	if (!_node->is_initialized) {
		printf("Error: Var %s is broken\n", _node->name.c_str());

		return;
	}

	if (is_ref){
		/// ���ʹ���ⲿ���ݵ�����
		_node->data = data;
		_node->has_data = true;

		return;
	}

	DataType *mem = _node->data.get();

	for (size_t i = 0; i < _node->getDataSize(); ++i) {
		mem[i] = data[i];
	}

	_node->has_data = true;
}

template<typename DataType>
void Var<DataType>::init(const std::initializer_list<DataType> &data){
	if (!_node->is_initialized) {
		printf("Error: Var %s is broken\n", _node->name.c_str());

		return;
	}

	DataType *mem = _node->data.get();

	int i = 0, max_i = _node->getDataSize();

	for (const auto &iter : data) {
		if (i > max_i) break;
		mem[i] = iter;
		++i;
	}

	_node->has_data = true;
}


template<typename DataType>
void init(const std::vector<DataType> &data) {
	if (!_node->is_initialized) {
		printf("Error: Var %s is broken\n", _node->name.c_str());

		return;
	}

	DataType *mem = _node->data.get();

	for (size_t i = 0; i < _node->getDataSize(); ++i) {
		mem[i] = data[i];
	}

	_node->has_data = true;
}

//====================================================


template<typename DataType>
inline opNode<DataType> * Var<DataType>::getNode() {
	return _node;
}

//====================================================


template<typename DataType>
inline opNode<DataType>& Var<DataType>::operator+(Var<DataType>& rhsObj) // ?
{
	opNode<DataType> &res = *_node + *rhsObj._node; 
	return res;
}