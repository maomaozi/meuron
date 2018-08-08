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

/// ��ģ�����͵Ļ��࣬Ϊ����ӿ�
/// ��Ҫ��һЩgetters�������ȡĳЩ�����޹�����
/// __Node �Ƕ��ڲ��ڵ���ͼ�ĳ���
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
	/// ����opNode�ϵĺ���ָ��
	using operator_mono_t = void (*)(const opNode<T> *, opNode<T> *);
	using operator_bino_t = void (*)(const opNode<T> *, const opNode<T> *, opNode<T> *);
	using data_type = T;

/// ���캯����
public:
	/// dummy�Ĺ��캯�����õ�ûʲô�õĿսڵ�
	opNode();

	/// �������죬���һ����ȫ��ͬ�Ľڵ㣬����ַ����ײ�����
	opNode(const opNode<T> &rhsObj);

	/// �ƶ�����
	opNode(opNode<T> &&rhsObj);

	///����ֵ��ʼ�������ڻ��һ�������ڵ㣬��Ҫ��ʽ�����ڵ�
	///�ӹ�initData���������ڴ��ͷ�
	opNode(T *initData, const std::vector<int> &initShape, const std::string &name);

	/// ���������ڹ�ͼʱ���Զ����ã����������󶨵��ڵ���
	opNode(opNode<T> * const lhsObj, operator_mono_t op);
	opNode(opNode<T> * const lhsObj, opNode<T> * const rhsObj, operator_bino_t op);

	virtual ~opNode();

/// ��������
public:
	/// ��˫Ŀ������������������ϣ��Լ�����һ�������������½ڵ�
	opNode<T> &bind(opNode<T> &rhs, operator_bino_t op);
	
	/// ����������Լ����ϣ��������½ڵ�
	opNode<T> &bind(operator_mono_t op);
	
	/// ִ�о������ 
	void exec();
	/// �����㵱ǰ�ڵ㣬�����ݹ����ǰ��
	void execHere();

	void reset();

	/// ��ӡ��Ϣ
	void repr();

/// Getters
public:
	const std::string getName() const;
	const std::vector<int> getShape() const;
	size_t getDataSize() const;

	// �����汾
	const __Node *fetchlhsParent() const;
	const __Node *fetchrhsParent() const;
	const std::unordered_set<std::shared_ptr<__Node>> &fetchNext() const;

	__Node *fetchlhsParent();
	__Node *fetchrhsParent();
	std::unordered_set<std::shared_ptr<__Node>> &fetchNext();

/// Setters
public:
	void setName(const std::string &);

/// ���ص������
public:
	T& operator=(opNode<T> rhsObj);
	bool operator==(const opNode<T> &rhsObj);
	void operator()();

	template <typename U>
	opNode<U> &&cast(opNode<T> &rhsObj);

	opNode<T> &operator+(opNode<T> &rhsObj);

/// ���߷���
private:
	void swap(opNode<T> &lhsNode, opNode<T> &rhsNode);

/// �洢��Ա
private:
	opNode<T> * lhs = nullptr;
	opNode<T> * rhs = nullptr;

	std::shared_ptr<T> data = nullptr;			// ָ�����������ݵ�����ָ��
	//std::shared_ptr<opNode<T>> next = nullptr;	// ����ָ����һ���ڵ������ָ�룬��ֹ����
	/* ���ǵ����ܳ����ж���������������Ը�Ϊset */
	std::unordered_set<std::shared_ptr<__Node>> next;

	operator_mono_t func_mono = nullptr;
	operator_bino_t func_bino = nullptr;

	size_t dataSize = 0;					// �����ݳ��ȣ�����shape�ƶϣ�����������ʵ�ʳ���

/// ����
private:
	std::vector<int> shape;					// ��״ N, C, W, H
	std::string name;						// ��ʶ��
	bool has_data = false;					// �Ƿ���һ������������Ѿ����㵱ǰ�ڵ�
	bool is_initialized = false;			// �Ƿ��Ѿ���ʼ����ָ���Զ��������ݳ�ʼ����������ͼ�ڲ��ڵ㣬ʼ����Ϊ�Ѿ���ʼ����
};