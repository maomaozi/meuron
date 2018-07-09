// ops_bind.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include "tensor.cpp"
#include "opnode.cpp"
#include "ops.cpp"
//#include "graph.cpp"

int main()
{
	/*
	INT8_t *v1 = new INT8_t[5]{ 1,2,3,4,5 };
	INT8_t *v2 = new INT8_t[5]{ 1,2,3,4};

	opNode<INT8_t> a(v1, { 5 }, "a");
	opNode<INT8_t> b(v2, { 4 }, "b");
	opNode<INT8_t> c(v2, { 4 }, "c");

	opNode<INT8_t> d = a + b;
	opNode<INT8_t> f = a + c;
	opNode<INT8_t> e = d + f + a;

	e();
	*/

	//user
	
	auto graph = Graph();

	auto a = graph.get_var<INT8_t>({ 2, 3 }, "a");
	auto b = graph.get_var<INT8_t>({ 2, 3 }, "b");
	auto c = graph.get_var<INT8_t>({ 2, 4 }, "c");

	Var<INT8_t> d = a + b;
	Var<INT8_t> e = b + c;
	Var<INT8_t> f = e + c;
	Var<INT8_t> g = f + f;

	a.init(
	{ 
		1, 2, 3, 
		4, 5, 6,
	});

	b.init(
	{ 
		6, 5, 4,
		3, 2, 1,
	});
	c.init(
	{
		1, 2, 3,
		4, 5, 6,
	});

	graph.finalize();
	graph.run(&g);

	g.getNode()->repr();
}

