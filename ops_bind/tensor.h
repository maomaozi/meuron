#pragma once

#include "types.h"

class __Node;

template <typename T>
class opNode;

/// ��ģ������������������Ͳ���
/// Tensor �Ƕ��û���ͼ���͵ĳ���
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
	/// Var���н�����ʽת������������Ϊһ���������㷵��ֵ�İ�װ��ʹ��
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
	/// ����Ϊͳһ��װ��ʱ��Ӧ��ת������
	//opNode<DataType> &operator+(Var<DataType> &rhsObj);
	Var<DataType> operator+(Var<DataType> &rhsObj);

private:
	/// ��ָͨ���������ʶ���
	opNode<DataType>* _node;
	/// �������ָ�����������½����������������������ط�����������ô����������
	/// ���� ��var��Ϊ��װ��������������ͼ���������еķ���ֵʱ
	/// �����Ѿ���opTree��ά���ˣ���������û��ֵ
	/// ��һ������� �����������ڵ㣬��ô�ڴ�Ӧ��Var���Լ�ά������������ζ�����ָͨ����з���
	/// ����ָ����˽�еģ��������Ŷ������������ܴ��ڲ���ȷ������
	std::shared_ptr<opNode<DataType>> _saver;
};


template <typename DataType>
class Const : public Tensor{
	Const() = delete;		///Const��������ʼ��
	Const (DataType *data);
	Const(std::initializer_list<DataType> data);

public:
	virtual opNode<DataType> *getNode();

private:
	opNode<DataType>* _node;
};