#include "CUDARasterBuffer.h"

#include "Utility/CUDA/CudaHelper.cuh"

#include "Raster Engine\RasterBackend.h"
#include <cuda_gl_interop.h>

CUDARasterBuffer::CUDARasterBuffer(CUDADevice* device,uint size) {

	cudaSetDevice(device->order);

	rasterBuffer = RasterBackend::CreateBuffer(size);

	if (RasterBackend::backend == OpenGL) {

		OpenGLBuffer* oglBuffer = static_cast<OpenGLBuffer*>(rasterBuffer);

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

CUDARasterBuffer::~CUDARasterBuffer() {


}
