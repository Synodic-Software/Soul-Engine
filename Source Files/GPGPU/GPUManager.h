#pragma once

#include "GPURasterBuffer.h"
#include "Metrics.h"

enum GPUBackend {CUDA, OpenCL};

namespace GPUManager {

	void ExtractDevices();

	GPURasterBuffer* CreateRasterBuffer(int,uint);
	int GetBestGPU();
}