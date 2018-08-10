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

template<class T>
__global__ void mat_add_kernel2(T* a, T* b, T* c, size_t size, const int offset)
{
	unsigned int i = blockIdx.x * blockDim.x * 4 + threadIdx.x;
	unsigned int k = i + offset;

	if (k < size) c[i] = a[k] + b[k];
	if (k + blockDim.x < size)	c[i + blockDim.x] = a[k + blockDim.x] + b[k + blockDim.x];
	if (k + 2 * blockDim.x < size)	c[i + blockDim.x * 2] = a[k + blockDim.x * 2] + b[k + blockDim.x * 2];
	if (k + 3 * blockDim.x < size)	c[i + blockDim.x * 3] = a[k + blockDim.x * 3] + b[k + blockDim.x * 3];
}




__host__ void gpu_mat_sum(char* lv, char* rv, char* res, size_t data_size)
{
	dim3 block(512);
	dim3 grid((data_size + block.x - 1) / block.x);
	mat_add_kernel<char> <<< grid, block >>> (lv, rv, res, data_size);
}


__host__ void gpu_mat_sum2(char* lv, char* rv, char* res, size_t data_size, const int offset)
{
	dim3 block(512);
	dim3 grid((data_size + block.x - 1) / block.x);
	mat_add_kernel2<char> <<< grid, block >>> (lv, rv, res, data_size, offset);
}