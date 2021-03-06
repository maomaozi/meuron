// ops_bind.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"

#include "tensor.cpp"
#include "opnode.cpp"
#include "ops.cpp"
#include <vector>

int main()
{

	auto graph = Graph();

	auto a = graph.get_var<INT8_t>({ 2, 3 }, "a");
	auto b = graph.get_var<INT8_t>({ 2, 3 }, "b");
	auto c = graph.get_var<INT8_t>({ 2, 3 }, "c");

	//std::vector<int8_t> h_a;
	//h_a.resize(1 << 24);

	//std::vector<int8_t> h_b;
	//h_b.resize(1 << 24);

	auto d = a + b;
	auto e = b + c;
	auto f = e + c;
	auto g = f + f;
	auto h = ops::sum_op::mat_sum<INT8_t>(a, b);

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
	graph.run(&h);

	h.getNode()->repr();
}

