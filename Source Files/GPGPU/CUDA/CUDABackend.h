#pragma once

#include <vector>

#include "Metrics.h"
#include "GPGPU\CUDA\CUDARasterBuffer.h"

namespace CUDABackend {

	void ExtractDevices(std::vector<int>&);

	void InitThread();

	int GetCoreCount();

	int GetWarpSize();

	int GetSMCount();

	int GetBlockHeight();

	void Terminate();

}