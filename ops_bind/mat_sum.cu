#include "stdafx.h"

template<class T> 
__global__ void vecaddOnDevice(T* a, T* b, T* c, const int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) c[i] = a[i] + b[i];
}


template<class T>
__host__ void gpuMatAdd(T* lv, T* rv, T* res, size_t dataSize)
{
	dim3 block(512);
	dim3 grid((dataSize + block.x - 1) / block.x);
	vecaddOnDevice<T> <<<grid, block >>> (lv, rv, res, dataSize);
}
