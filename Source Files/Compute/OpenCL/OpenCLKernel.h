#pragma once
#include "Metrics.h"
#include "Compute\GPUKernel.h"

template<class T>
class OpenCLKernel :public GPUKernel<T> {

public:

	OpenCLKernel()
		: GPUKernel() {

	}

	/* Destructor. */
	~OpenCLKernel() {}


protected:

private:

};