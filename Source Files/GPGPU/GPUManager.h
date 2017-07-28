#pragma once

#include "Metrics.h"

#include "CUDA\CUDABuffer.h"
#include "OpenCL\OpenCLBuffer.h"

#include "CUDA\CUDARasterBuffer.h"
#include "OpenCL\OpenCLRasterBuffer.h"

#include <memory>
#include <vector>

/* . */
namespace GPUManager {

	namespace detail {

		/* The devices */
		extern std::vector<std::unique_ptr<GPUDevice>> devices;

	}

	/* Extracts the devices. */
	void ExtractDevices();
	/* Destroys the devices. */
	void DestroyDevices();

	/* Initializes the thread. */
	void InitThread();

	/*
	 *    Creates raster buffer.
	 *    @param	parameter1	The first parameter.
	 *    @param	parameter2	The second parameter.
	 *    @return	Null if it fails, else the new raster buffer.
	 */

	template<typename T>
	GPURasterBuffer<T>* CreateRasterBuffer(int GPU, uint size) {

		GPURasterBuffer<T>* buffer;

		if (detail::devices[GPU]->api == CUDA) {
			buffer = new CUDARasterBuffer<T>(*detail::devices[GPU], size);
		}
		else {
			buffer = new OpenCLRasterBuffer<T>(*detail::devices[GPU], size);
		}
		return buffer;
	}

	/*
	 *    Creates a buffer.
	 *    @param	parameter1	The first parameter.
	 *    @param	parameter2	The second parameter.
	 *    @return	Null if it fails, else the new buffer.
	 */
	 
	template<typename T>
	GPUBuffer<T>* CreateBuffer(int GPU, uint size) {

		GPUBuffer<T>* buffer;

		if (detail::devices[GPU]->api == CUDA) {
			buffer = new CUDABuffer<T>(*detail::devices[GPU], size);
		}
		else {
			buffer = new OpenCLBuffer<T>(*detail::devices[GPU], size);
		}
		return buffer;
	}

	/*
	 *    Gets best GPU.
	 *    @return	The best GPU.
	 */

	int GetBestGPU();
}