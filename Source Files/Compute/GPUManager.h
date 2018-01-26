#pragma once


#include "CUDA/CUDABackend.h"
#include "OpenCL/OpenCLBackend.h"

#include <vector>

/* . */
namespace GPUManager {

	namespace detail {

		/* The devices */
		extern std::vector<GPUDevice> devices;
		extern CUDABackend cudaBackend;
		extern OpenCLBackend openCLBackend;
	}

	/* Extracts the devices. */
	void ExtractDevices();
	/* Destroys the devices. */
	void DestroyDevices();

	/* Initializes the thread. */
	void InitThread();

	/*
	 *    Gets best GPU.
	 *    @return	The best GPU.
	 */

	GPUDevice GetBestGPU();
}
