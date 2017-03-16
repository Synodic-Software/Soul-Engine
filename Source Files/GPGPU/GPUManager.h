#pragma once

#include "GPURasterBuffer.h"


enum GPUBackend {CUDA, OpenCL};

namespace GPUManager {

	void ExtractDevices();

	GPURasterBuffer* CreateRasterBuffer();
}