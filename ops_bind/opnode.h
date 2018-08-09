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
	virtual const std::string &get_name() const = 0;
	virtual const std::vector<int> &get_shape() const = 0;
	virtual bool get_initialized() const = 0;
	virtual bool get_calculated() const = 0;
	virtual size_t get_datasize() const = 0;

	virtual const __Node *get_lhs_parent() const = 0;
	virtual const __Node *fetch_rhs_parent() const = 0;
	virtual const std::unordered_set<std::shared_ptr<__Node>> &fetchNext() const = 0;

	virtual __Node* get_lhs_parent() = 0;
	virtual __Node* fetch_rhs_parent() = 0;
	virtual std::unordered_set<std::shared_ptr<__Node>> &fetchNext() = 0;

	virtual void exec_here() = 0;
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
	void exec_here();

	void reset();

	/// ��ӡ��Ϣ
	void repr();

/// Getters
public:

	std::shared_ptr<T> &opNode<T>::get_data();
	__Node *get_lhs_parent();
	__Node *fetch_rhs_parent();
	std::unordered_set<std::shared_ptr<__Node>> &fetchNext();

	// �����汾
	const	std::string &get_name()			const;
	const	std::vector<int> &get_shape()	const;
	bool	get_initialized()				const;
	size_t	get_datasize()					const;
	bool	get_calculated()				const;

	const	__Node *get_lhs_parent()		const;
	const	__Node *fetch_rhs_parent()		const;
	const	std::unordered_set<std::shared_ptr<__Node>> &fetchNext()	const;

	const	std::shared_ptr<T> &opNode<T>::fetch_data()					const;

/// Setters
public:
	void set_name(const std::string &);
	void set_shape(const std::vector<int> &);
	void set_datasize(size_t);
	void set_initialized(bool);
	void set_calculated(bool);

/// ���ص������
public:
	T& operator=(opNode<T> rhsObj);
	bool operator==(const opNode<T> &rhsObj);
	void operator()();

	//template <typename U>
	//opNode<U> &&cast(opNode<T> &rhsObj);

	opNode<T> &operator+(opNode<T> &rhsObj);

/// ���߷���
private:
	void swap(opNode<T> &lhsNode, opNode<T> &rhsNode);

/// �洢��Ա
private:
	opNode<T> *lhs = nullptr;
	opNode<T> *rhs = nullptr;

	std::shared_ptr<T> data = nullptr;					// ָ�����������ݵ�����ָ��
	//std::shared_ptr<opNode<T>> next = nullptr;		// ����ָ����һ���ڵ������ָ�룬��ֹ����
	/* ���ǵ����ܳ����ж���������������Ը�Ϊset */
	std::unordered_set<std::shared_ptr<__Node>> next;

	operator_mono_t func_mono = nullptr;
	operator_bino_t func_bino = nullptr;

	size_t data_size = 0;					// �����ݳ��ȣ�����shape�ƶϣ�����������ʵ�ʳ���

/// ����
private:
	std::vector<int> shape;					// ��״ N, C, W, H
	std::string name;						// ��ʶ��
	bool is_calculated = false;				// �Ƿ���һ������������Ѿ����㵱ǰ�ڵ�
	bool is_initialized = false;			// �Ƿ��Ѿ���ʼ����ָ���Զ��������ݳ�ʼ����������ͼ�ڲ��ڵ㣬ʼ����Ϊ�Ѿ���ʼ����
};
