#pragma once

#include "Metrics.h"

enum GPUBackend { CUDA, OpenCL };

class GPUDevice {

public:
	GPUDevice(uint);
	virtual ~GPUDevice()=0;



	GPUBackend api;
	int order;
protected:

private:

};