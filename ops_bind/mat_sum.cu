#include "stdafx.h"

__global__ void vecaddOnDevice(float* a, float* b, float* c, const int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) c[i] = a[i] + b[i];
}