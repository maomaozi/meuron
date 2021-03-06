// stdafx.h : 标准系统包含文件的包含文件，
// 或是经常使用但不常更改的
// 特定于项目的包含文件
//

#pragma once

#define USE_CUDA

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>
#include <stdint.h>


// TODO: 在此处引用程序需要的其他头文件
#include <map>
#include <vector>
#include <unordered_set>
#include <functional>
#include <vector>
#include <memory>
#include <iostream>
#include <type_traits>

#ifdef USE_CUDA

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define CHECK(call) 														\
{                   														\
	const cudaError_t error = call;											\
	if(error != cudaSuccess)												\
	{																		\
		printf("Error: %s:%d, ", __FILE__, __LINE__);						\
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));	\
	}																		\
}																		\

#endif // USE_CUDA


#include "graph.h"
#include "opnode.h"
#include "types.h"
#include "ops.h"
#include "tensor.h"
#include "utils.h"
