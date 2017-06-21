#pragma once

#include "GPURasterBuffer.h"
#include "Metrics.h"

namespace GPUManager {

	void ExtractDevices();
	void DestroyDevices();

	void InitThread();

	GPURasterBuffer* CreateRasterBuffer(int,uint);
	GPUBuffer* CreateBuffer(int, uint);
	int GetBestGPU();
}