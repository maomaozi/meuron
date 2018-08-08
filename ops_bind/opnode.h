#pragma once

#include "stdafx.h"


namespace ops {
	class op;
}


class Tensor;


template <typename T>
class Var;


template <typename T>
class Mouth;


template <typename T>
class Const;

/// 无模板类型的基类，为纯虚接口
/// 主要放一些getters，方便获取某些类型无关属性
/// __Node 是对内部节点视图的抽象
class __Node {
public:
	virtual const std::string getName() const = 0;
	virtual const std::vector<int> getShape() const = 0;
	virtual size_t getDataSize() const = 0;

	virtual const __Node *fetchlhsParent() const = 0;
	virtual const __Node *fetchrhsParent() const = 0;
	virtual const std::unordered_set<std::shared_ptr<__Node>> &fetchNext() const = 0;

	virtual __Node* fetchlhsParent() = 0;
	virtual __Node* fetchrhsParent() = 0;
	virtual std::unordered_set<std::shared_ptr<__Node>> &fetchNext() = 0;

	virtual void execHere() = 0;
};


template <typename T>
class opNode : public __Node {
public:
	friend class ops::op;

	friend class Tensor;
	friend class Var<T>;
	friend class Mouth<T>;
	friend class Const<T>;
	
public:
	/// 绑定在opNode上的函数指针
	using operator_mono_t = void (*)(const opNode<T> *, opNode<T> *);
	using operator_bino_t = void (*)(const opNode<T> *, const opNode<T> *, opNode<T> *);
	using data_type = T;

/// 构造函数们
public:
	/// dummy的构造函数，得到没什么用的空节点
	opNode();

	/// 拷贝构造，获得一个完全相同的节点，按地址共享底部数据
	opNode(const opNode<T> &rhsObj);

	/// 移动构造
	opNode(opNode<T> &&rhsObj);

	///按照值初始化，用于获得一个完整节点，需要显式命名节点
	///接管initData，负责其内存释放
	opNode(T *initData, const std::vector<int> &initShape, const std::string &name);

	/// 下面两个在构图时被自动调用，将操作符绑定到节点上
	opNode(opNode<T> * const lhsObj, operator_mono_t op);
	opNode(opNode<T> * const lhsObj, opNode<T> * const rhsObj, operator_bino_t op);

	virtual ~opNode();

/// 操作方法
public:
	/// 绑定双目运算符到两个操作数上（自己和另一个），并生成新节点
	opNode<T> &bind(opNode<T> &rhs, operator_bino_t op);
	
	/// 绑定运算符在自己身上，并产生新节点
	opNode<T> &bind(operator_mono_t op);
	
	/// 执行具体计算 
	void exec();
	/// 仅计算当前节点，而不递归计算前驱
	void execHere();

	void reset();

	/// 打印信息
	void repr();

/// Getters
public:
	const std::string getName() const;
	const std::vector<int> getShape() const;
	size_t getDataSize() const;

	// 常量版本
	const __Node *fetchlhsParent() const;
	const __Node *fetchrhsParent() const;
	const std::unordered_set<std::shared_ptr<__Node>> &fetchNext() const;

	__Node *fetchlhsParent();
	__Node *fetchrhsParent();
	std::unordered_set<std::shared_ptr<__Node>> &fetchNext();

/// Setters
public:
	void setName(const std::string &);

/// 重载的运算符
public:
	T& operator=(opNode<T> rhsObj);
	bool operator==(const opNode<T> &rhsObj);
	void operator()();

	template <typename U>
	opNode<U> &&cast(opNode<T> &rhsObj);

	opNode<T> &operator+(opNode<T> &rhsObj);

/// 工具方法
private:
	void swap(opNode<T> &lhsNode, opNode<T> &rhsNode);

/// 存储成员
private:
	opNode<T> * lhs = nullptr;
	opNode<T> * rhs = nullptr;

	std::shared_ptr<T> data = nullptr;			// 指向所保存数据的智能指针
	//std::shared_ptr<opNode<T>> next = nullptr;	// 保存指向下一个节点的智能指针，防止析构
	/* 考虑到可能出现有多个后继运算符，所以改为set */
	std::unordered_set<std::shared_ptr<__Node>> next;

	operator_mono_t func_mono = nullptr;
	operator_bino_t func_bino = nullptr;

	size_t dataSize = 0;					// 总数据长度，根据shape推断，不代表数据实际长度

/// 属性
private:
	std::vector<int> shape;					// 形状 N, C, W, H
	std::string name;						// 标识名
	bool has_data = false;					// 是否在一次推理过程中已经计算当前节点
	bool is_initialized = false;			// 是否已经初始化（指属性而不是数据初始化，对于流图内部节点，始终认为已经初始化）
};