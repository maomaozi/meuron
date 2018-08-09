#include "stdafx.h"

#include "mat_sum.h"

template<class T> 
__global__ void mat_add_kernel(T *a, T *b, T *c, const int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < size) 
	{
		c[i] = a[i] + b[i];
	}
}


__host__ void gpu_mat_sum(char* lv, char* rv, char* res, size_t data_size)
{
	dim3 block(512);
	dim3 grid((data_size + block.x - 1) / block.x);
	mat_add_kernel<char> <<< grid, block >>> (lv, rv, res, data_size);
}