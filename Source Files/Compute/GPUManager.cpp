#include "GPUManager.h"

#include "GPUDevice.h"
#include "CUDA\CUDADevice.cuh"
#include "OpenCL\OpenCLDevice.h"

namespace GPUManager {

	/* Only run the function if there is an available. */

	namespace detail {

		/* The devices */
		std::vector<GPUDevice> devices;

		CUDABackend cudaBackend;
		OpenCLBackend openCLBackend;

	}

	void ExtractDevices() {
		std::vector<CUDADevice> cudaDevices;
		detail::cudaBackend.ExtractDevices(cudaDevices);

		int cudaCounter = 0;
		int clCounter = 0;

		for (auto &device : cudaDevices) {
			detail::devices.emplace_back( new CUDADevice(device) );
			++cudaCounter;
		}

		std::vector<OpenCLDevice> openCLDevices;
		detail::openCLBackend.ExtractDevices(openCLDevices);

		for (auto &device : openCLDevices) {
			detail::devices.emplace_back(new OpenCLDevice(device));

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

	GPUDevice GetBestGPU() {
		return detail::devices[0];
	}


}