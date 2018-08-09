#include "stdafx.h"
#include "utils.h"



std::string vec_shape_to_str(const std::vector<int>& vec) 
{
	if (vec.empty()) 
	{
		return "(NULL)";
	}

	std::string res = "(";
	char tmp[10];
	for (size_t i = 0; i < vec.size(); ++i) 
	{
		itoa(vec[i], tmp, 10);
		res.append(tmp);
		if (i != vec.size() - 1) 
		{
			res.append(", ");
		}
	}
	res.append(")");

	return res;
}


size_t shape_to_size(const std::vector<int>& shape) 
{
	size_t dataSizeAccu = 1;

	for (const int &i : shape) 
	{
		if (i <= 0)
		{
			return -1;
		}
		dataSizeAccu *= i;
	}

	return dataSizeAccu;
}