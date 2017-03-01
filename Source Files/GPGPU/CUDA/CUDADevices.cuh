#pragma once

#include <vector>

namespace CUDAProperties{

	void ExtractDevices(std::vector<int>&);

	int GetCoreCount();

	int GetWarpSize();

	int GetSMCount();

	int GetBlockHeight();

	void Terminate();

}