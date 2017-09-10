#include "GPUManager.h"

#include "GPUDevice.h"
#include "CUDA\CUDADevice.h"
#include "OpenCL\OpenCLDevice.h"

namespace GPUManager {

	/* Only run the function if there is an available. */

	namespace detail {

		/* The devices */
		std::vector<std::unique_ptr<GPUDevice>> devices;

		CUDABackend cudaBackend;
		OpenCLBackend openCLBackend;

	}

	void ExtractDevices() {
		std::vector<GPUDevice> cudaDevices;
		detail::cudaBackend.ExtractDevices(cudaDevices);

		int cudaCounter = 0;
		int clCounter = 0;

		for (auto &var : cudaDevices) {
			detail::devices.emplace_back(new CUDADevice(cudaCounter));
			++cudaCounter;
		}

		std::vector<GPUDevice> openCLDevices;
		detail::openCLBackend.ExtractDevices(openCLDevices);

		for (auto &var : openCLDevices) {
			detail::devices.emplace_back(new OpenCLDevice(clCounter));

			++clCounter;
		}
	}

	/* Destroys the devices. */
	void DestroyDevices() {
		detail::devices.clear();
	}

	/* Initializes the thread. */
	void InitThread() {
		detail::cudaBackend.InitThread();
		detail::openCLBackend.InitThread();
	}

	/*
	 *    Gets best GPU.
	 *    @return	The best GPU.
	 */

	GPUDevice& GetBestGPU() {
		return *detail::devices[0].get();
	}


}