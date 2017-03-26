#pragma once

#include "GPURasterBuffer.h"
#include "Metrics.h"

namespace GPUManager {

	void ExtractDevices();

	GPURasterBuffer* CreateRasterBuffer(int,uint);
	int GetBestGPU();
}