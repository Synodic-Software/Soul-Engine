#pragma once

#include "Utility\CUDA\CudaHelper.cuh"
#include "thrust/device_vector.h"

class Managed
{
public:
	void *operator new(size_t len);
	void operator delete(void *ptr);
};
