// ops_bind.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include "tensor.cpp"
#include "opnode.cpp"
#include "ops.cpp"
//#include "graph.cpp"

int main()
{

	auto graph = Graph();

	auto a = graph.get_var<INT8_t>({ 2, 3 }, "a");
	auto b = graph.get_var<INT8_t>({ 2, 3 }, "b");
	auto c = graph.get_var<INT8_t>({ 2, 3 }, "c");

	auto d = a + b;
	auto e = b + c;
	auto f = e + c;
	auto g = f + f;
	auto h = a + b;

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

