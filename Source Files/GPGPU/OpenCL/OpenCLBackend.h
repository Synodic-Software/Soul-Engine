#pragma once

#include <vector>
#include <Metrics.h>
#include "GPGPU\OpenCL\OpenCLRasterBuffer.h"

namespace OpenCLBackend {

	void ExtractDevices(std::vector<int>&);

	void InitThread();

	int GetCoreCount();

	int GetWarpSize();

	int GetSMCount();

	int GetBlockHeight();

	void Terminate();

}