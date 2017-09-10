#pragma once

#include "Metrics.h"

#include "CUDA/CUDABackend.h"
#include "OpenCL/OpenCLBackend.h"

#include <memory>
#include <vector>

/* . */
namespace GPUManager {

	namespace detail {

		/* The devices */
		extern std::vector<std::unique_ptr<GPUDevice>> devices;
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
	 *    Transfer to device.
	 *    @tparam	T	Generic type parameter.
	 *    @param [in,out]	device	The device.
	 */

	template <typename T>
	void TransferToDevice(GPUDevice& device, std::vector<GPUBuffer<T>> buffer) {
		//TODO implement for each backend

		if (device.api == CUDA) {
			detail::cudaBackend.TransferToDevice(device, buffer);
		}
		else if (device.api == OpenCL) {
			detail::openCLBackend.TransferToDevice(device, buffer);
		}
	}

	/*
	 *    Transfer to host.
	 *    @tparam	T	Generic type parameter.
	 *    @param [in,out]	device	The device.
	 */

	template <typename T>
	void TransferToHost(GPUDevice& device, std::vector<GPUBuffer<T>>) {
		//TODO implement
	}

	/*
	 *    Gets best GPU.
	 *    @return	The best GPU.
	 */

	GPUDevice& GetBestGPU();
}
