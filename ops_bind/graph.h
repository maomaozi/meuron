#pragma once

#include "tensor.h"

/// ��ɶ�opNode���߲�εľۺϲ���
/// �����ڴ�ķ�������ݳ�ʼ��������״̬��ά���Լ��ڵ�֮���ϵ��ά��
/// �ڵ�Ĳ�ι�ϵ�Լ���������
/// Graph���û���ͼ���ڲ��ڵ���ͼ�Ľ���

class Graph {
public:
	Graph() {};

	/// ��ȡһ����̬������ڵ�
	template <typename DataType>
	Mouth<DataType> & get_mouth();

	/// ��ȡһ������
	template <typename DataType>
	Var<DataType> & get_var(const std::vector<int> shape, std::string name);

	///��ȡһ������
	template <typename DataType>
	Const<DataType> & get_const(const std::initializer_list<DataType> &initData);

	/// ���һ��ͼģ�����ռ���ǰ׼��
	void finalize();

	/// ��
	void run(Tensor *tensor);

private:
	/// ������ͼ�ļ�����ڵ�
	std::unordered_set<std::shared_ptr<Tensor>> entryNodes;

	///�����˲㼶ģ��
	std::vector<std::unordered_set<__Node *>> nodesLevels;

	///�����˽ڵ�node���㼶��ӳ��
	std::map<const __Node*, size_t> nodeLevelMap;
};


template<typename DataType>
inline Mouth<DataType>& Graph::get_mouth()
{
	std::shared_ptr<Tensor> ptr = new Mouth<DataType>();

	entryNodes.insert(ptr);

	return *ptr;
}

template<typename DataType>
inline Var<DataType>& Graph::get_var(const std::vector<int> shape, std::string name)
{
	std::shared_ptr<Tensor> ptr((Tensor *)new Var<DataType>(shape, name));

	entryNodes.insert(ptr);

	return *dynamic_cast<Var<DataType> *>(ptr.get());
}

template<typename DataType>
inline Const<DataType>& Graph::get_const(const std::initializer_list<DataType> &initData)
{
	std::shared_ptr<Tensor> ptr = new Const<DataType>(initData);

	entryNodes.insert(ptr);

	return *ptr;
}