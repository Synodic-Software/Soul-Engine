#include "GPUManager.h"

#include "CUDA/CUDADevices.cuh"
#include <vector>

std::vector<std::pair<int, GPUBackend>> devices;

namespace GPUManager {

	void ExtractDevices() {
		std::vector<int> cudaDevices;
		CUDAProperties::ExtractDevices(cudaDevices);

		for (auto &var : cudaDevices) {
			devices.push_back(std::make_pair(var,CUDA));
		}
	}


}