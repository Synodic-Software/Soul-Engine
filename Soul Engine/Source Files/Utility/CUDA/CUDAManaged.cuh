#pragma once

#include "Utility\CUDA\CudaHelper.cuh"

class Managed
{
public:
	void *operator new(size_t len);
	void operator delete(void *ptr);
};
