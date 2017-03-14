#include "CUDABackend.h"
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "Utility/CUDA/CudaHelper.cuh"

#include "Raster Engine\RasterBackend.h"
#include "Raster Engine\OpenGL\OpenGLBuffer.h"
#include <cuda_gl_interop.h>

static int deviceCount;
static cudaDeviceProp* deviceProp;

namespace CUDABackend {

	void ExtractDevices(std::vector<int>& devices) {
		cudaError_t error = cudaGetDeviceCount(&deviceCount);

		if (deviceCount == 0)
		{
			return;
		}

		deviceProp = new cudaDeviceProp[deviceCount];

		for (int dev = 0; dev < deviceCount; ++dev) {

			CudaCheck(cudaSetDevice(dev));
			CudaCheck(cudaGetDeviceProperties(&deviceProp[dev], dev));
			devices.push_back(dev);
		}

	}

	int GetCoreCount() {
		int device;
		CudaCheck(cudaGetDevice(&device));
		return _ConvertSMVer2Cores(deviceProp[device].major, deviceProp[device].minor) * deviceProp[device].multiProcessorCount;
	}

	int GetSMCount() {
		int device;
		CudaCheck(cudaGetDevice(&device));
		return deviceProp[device].multiProcessorCount;
	}

	int GetWarpSize() {
		int device;
		CudaCheck(cudaGetDevice(&device));

		return deviceProp[device].warpSize;
	}

	int GetBlockHeight() {
		int device;
		CudaCheck(cudaGetDevice(&device));

		return _ConvertSMVer2Cores(deviceProp[device].major, deviceProp[device].minor) / GetWarpSize();
	}

	void Terminate() {
		CudaCheck(cudaDeviceReset());
	}

	void CreateRasterBuffer(uint size) {

		Buffer* rasterBuffer = RasterBackend::CreateBuffer(size);

		if (RasterBackend::backend == OpenGL) {

			OpenGLBuffer* oglBuffer = static_cast<OpenGLBuffer*>(rasterBuffer);

			struct cudaGraphicsResource *cudaBuffer;
			glm::vec4 *bufferData;

			CudaCheck(cudaGraphicsGLRegisterBuffer(&cudaBuffer
				, oglBuffer->GetBufferID()
				, cudaGraphicsRegisterFlagsWriteDiscard));


			CudaCheck(cudaGraphicsMapResources(1, &cudaBuffer, 0));
			size_t num_bytes;
			CudaCheck(cudaGraphicsResourceGetMappedPointer((void **)&bufferData, &num_bytes,
				cudaBuffer));
		}
		else {
			//TODO
			//Vulkan Stuff


		}



	}

}