#pragma once

#include "tensor.h"

/// 完成对opNode更高层次的聚合操作
/// 包括内存的分配和数据初始化，计算状态的维护以及节点之间关系的维护
/// 节点的层次关系以及并发计算
/// Graph是用户视图和内部节点视图的交接

class Graph {
public:
	Graph() {};

	/// 获取一个动态数据入口点
	template <typename DataType>
	Mouth<DataType> & get_mouth();

	/// 获取一个变量
	template <typename DataType>
	Var<DataType> & get_var(const std::vector<int> shape, std::string name);

	///获取一个常量
	template <typename DataType>
	Const<DataType> & get_const(const std::initializer_list<DataType> &initData);

	/// 完成一个图模型最终计算前准备
	void finalize();

	/// 算
	void run(Tensor *tensor);

private:
	/// 保存了图的计算入口点
	std::unordered_set<std::shared_ptr<Tensor>> entryNodes;

	///保存了层级模型
	std::vector<std::unordered_set<__Node *>> nodesLevels;

	///保存了节点node到层级的映射
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