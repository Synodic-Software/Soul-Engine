#pragma once

#include "Utility\CUDA\HelperClasses.cuh"

class Managed
{
public:
	void *operator new(size_t len);
	void operator delete(void *ptr);
};
