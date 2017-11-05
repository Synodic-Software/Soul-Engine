#pragma once
#include "Metrics.h"
#include "GPGPU\GPUKernel.h"

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