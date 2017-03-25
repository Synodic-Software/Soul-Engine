#include "GPUManager.h"

#include "CUDA\CUDABackend.h"
#include "OpenCL\OpenCLBackend.h"

#include "CUDA\CUDARasterBuffer.h"
#include "OpenCL\OpenCLRasterBuffer.h"

#include <vector>

std::vector<std::pair<int, GPUBackend>> devices;

namespace GPUManager {

	//Only run the function if there is an available 



	void ExtractDevices() {
		std::vector<int> cudaDevices;
		CUDABackend::ExtractDevices(cudaDevices);

		for (auto &var : cudaDevices) {
			devices.push_back(std::make_pair(var, CUDA));
		}

		std::vector<int> openCLDevices;
		OpenCLBackend::ExtractDevices(openCLDevices);

		for (auto &var : openCLDevices) {
			devices.push_back(std::make_pair(var, OpenCL));
		}
	}


	GPURasterBuffer* CreateRasterBuffer(int GPU, uint size) {

		GPURasterBuffer* buffer;

		if (devices[GPU].second == CUDA) {
			//cudaSetDevice(1);
			buffer = new CUDARasterBuffer(size);
		}
		else {
			buffer = new OpenCLRasterBuffer();
		}
		return buffer;
	}

	int GetBestGPU() {
		return 0;
	}


}