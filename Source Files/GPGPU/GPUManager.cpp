#include "GPUManager.h"

#include "CUDA\CUDABackend.h"
#include "OpenCL\OpenCLBackend.h"

#include "CUDA\CUDABuffer.h"
#include "OpenCL\OpenCLBuffer.h"

#include "CUDA\CUDARasterBuffer.h"
#include "OpenCL\OpenCLRasterBuffer.h"

#include "GPUDevice.h"
#include "CUDA\CUDADevice.h"
#include "OpenCL\OpenCLDevice.h"

#include <vector>
#include <memory>

/* The devices */
std::vector<std::unique_ptr<GPUDevice>> devices;

namespace GPUManager {

	/* Only run the function if there is an available. */



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

	/* Destroys the devices. */
	void DestroyDevices() {
		devices.clear();
	}


	/* Initializes the thread. */
	void InitThread() {
		CUDABackend::InitThread();
		OpenCLBackend::InitThread();
	}

	/*
	 *    Creates raster buffer.
	 *    @param	GPU 	The GPU.
	 *    @param	size	The size.
	 *    @return	Null if it fails, else the new raster buffer.
	 */

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

	/*
	 *    Creates a buffer.
	 *    @param	GPU 	The GPU.
	 *    @param	size	The size.
	 *    @return	Null if it fails, else the new buffer.
	 */

	GPUBuffer* CreateBuffer(int GPU, uint size) {

		GPUBuffer* buffer;

		if (devices[GPU]->api == CUDA) {
			buffer = new CUDABuffer(static_cast<CUDADevice*>(devices[GPU].get()), size);
		}
		else {
			buffer = new OpenCLBuffer(static_cast<OpenCLDevice*>(devices[GPU].get()), size);
		}
		return buffer;
	}

	/*
	 *    Gets best GPU.
	 *    @return	The best GPU.
	 */

	int GetBestGPU() {
		return 0;
	}


}