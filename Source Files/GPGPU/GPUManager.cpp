#include "GPUManager.h"

#include "CUDA\CUDABackend.h"
#include "OpenCL\OpenCLBackend.h"

#include "CUDA\CUDARasterBuffer.h"
#include "OpenCL\OpenCLRasterBuffer.h"

#include "GPUDevice.h"
#include "CUDA\CUDADevice.h"
#include "OpenCL\OpenCLDevice.h"

#include <vector>
#include <memory>

std::vector<std::unique_ptr<GPUDevice>> devices;

namespace GPUManager {

	//Only run the function if there is an available 



	void ExtractDevices() {
		std::vector<int> cudaDevices;
		CUDABackend::ExtractDevices(cudaDevices);

		for (auto &var : cudaDevices) {
			devices.emplace_back(new CUDADevice(var));
		}

		std::vector<int> openCLDevices;
		OpenCLBackend::ExtractDevices(openCLDevices);

		for (auto &var : openCLDevices) {
			devices.emplace_back(new OpenCLDevice(var));
		}
	}

	void InitThread() {
		CUDABackend::InitThread();
		OpenCLBackend::InitThread();
	}


	GPURasterBuffer* CreateRasterBuffer(int GPU, uint size) {

		GPURasterBuffer* buffer;

		if (devices[GPU]->api == CUDA) {
			buffer = new CUDARasterBuffer(static_cast<CUDADevice*>(devices[GPU].get()),size);
		}
		else {
			buffer = new OpenCLRasterBuffer(static_cast<OpenCLDevice*>(devices[GPU].get()), size);
		}
		return buffer;
	}

	int GetBestGPU() {
		return 0;
	}


}