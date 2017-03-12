#pragma once

#include <vector>

#include "Metrics.h"

namespace CUDABackend {

	void ExtractDevices(std::vector<int>&);

	int GetCoreCount();

	int GetWarpSize();

	int GetSMCount();

	int GetBlockHeight();

	void Terminate();

	void CreateRasterBuffer(uint size);

}