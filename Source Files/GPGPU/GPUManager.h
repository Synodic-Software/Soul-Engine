#pragma once

#include "GPURasterBuffer.h"
#include "Metrics.h"

namespace GPUManager {

	void ExtractDevices();
	void DestroyDevices();

	void InitThread();

	GPURasterBuffer* CreateRasterBuffer(int,uint);
	int GetBestGPU();
}