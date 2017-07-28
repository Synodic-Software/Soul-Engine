#include "GPUManager.h"

#include "CUDA\CUDABackend.h"
#include "OpenCL\OpenCLBackend.h"

#include "GPUDevice.h"
#include "CUDA\CUDADevice.h"
#include "OpenCL\OpenCLDevice.h"

namespace GPUManager {

	/* Only run the function if there is an available. */

	namespace detail {

		/* The devices */
		std::vector<std::unique_ptr<GPUDevice>> devices;

	}

	void ExtractDevices() {
		std::vector<int> cudaDevices;
		CUDABackend::ExtractDevices(cudaDevices);

		for (auto &var : cudaDevices) {
			detail::devices.emplace_back(new CUDADevice(var));
		}

		std::vector<int> openCLDevices;
		OpenCLBackend::ExtractDevices(openCLDevices);

		for (auto &var : openCLDevices) {
			detail::devices.emplace_back(new OpenCLDevice(var));
		}
	}

	/* Destroys the devices. */
	void DestroyDevices() {
		detail::devices.clear();
	}


	/* Initializes the thread. */
	void InitThread() {
		CUDABackend::InitThread();
		OpenCLBackend::InitThread();
	}

	/*
	 *    Gets best GPU.
	 *    @return	The best GPU.
	 */

	int GetBestGPU() {
		return 0;
	}


}