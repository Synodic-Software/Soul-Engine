#pragma once
#include "Core/Utility/Types.h"
#include "Parallelism/Compute/GPUKernel.h"

template<class T>
class OpenCLKernel :public GPUKernel<T> {

public:

	OpenCLKernel()
		: GPUKernel<T>() {

	}

	/* Destructor. */
	~OpenCLKernel() {}


protected:

private:

};
