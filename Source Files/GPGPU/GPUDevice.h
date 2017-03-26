#pragma once

#include "Metrics.h"

enum GPUBackend { CUDA, OpenCL };

class GPUDevice {

public:
	GPUDevice(uint);
	~GPUDevice();

	GPUBackend api;
	int order;
protected:

private:

};