#pragma once

#include "types.h"

class __Node;

template <typename T>
class opNode;

/// 无模板参数基类用来做类型擦除
/// Tensor 是对用户视图类型的抽象
class Tensor {
public:
	Tensor() {};
	virtual ~Tensor() {};

public:
	virtual dataType getType() { return UNKOWN; };
	virtual __Node *getNode() = 0;

protected:
	dataType realType;
	std::vector<int> _shape;
};


template <typename DataType>
class Mouth : public Tensor {
public:
	void feed(DataType *data);
	void feed(std::initializer_list<DataType> data);

public:
	virtual opNode<DataType> *getNode();

private:
	opNode<DataType>* _node;
};


template <typename DataType>
class Var : public Tensor {
public:
	/// Var具有接受隐式转换的能力，作为一个接受运算返回值的包装器使用
	Var(opNode<DataType> &node);

	Var(Var<DataType> &rhsVar);

	Var(Var<DataType> &&rhsVar);

	Var(const std::vector<int> &initShape, std::string name);

public:
	void init(DataType *data, bool is_ref=false);
	void init(const std::initializer_list<DataType> &data);
	void init(const std::vector<DataType> &data);

public:
	virtual opNode<DataType> *getNode();

public:
	/// 当作为统一包装器时，应该转发计算
	//opNode<DataType> &operator+(Var<DataType> &rhsObj);
	Var<DataType> operator+(Var<DataType> &rhsObj);

private:
	/// 普通指针用来访问对象
	opNode<DataType>* _node;
	/// 这个智能指针用来保存新建对象，如若对象是在其他地方被创建，那么他将不启用
	/// 例如 当var作为包装器仅仅用来接受图构建过程中的返回值时
	/// 对象已经在opTree中维护了，所以这里没有值
	/// 另一种情况， 如果对象是入口点，那么内存应该Var类自己维护，但无论如何都从普通指针进行访问
	/// 由于指针是私有的，所以随着对象析构不可能存在不正确的引用
	std::shared_ptr<opNode<DataType>> _saver;
};


template <typename DataType>
class Const : public Tensor{
	Const() = delete;		///Const对象必须初始化
	Const (DataType *data);
	Const(std::initializer_list<DataType> data);

public:
	virtual opNode<DataType> *getNode();

private:
	opNode<DataType>* _node;
};