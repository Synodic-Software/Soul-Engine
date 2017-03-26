#include "CUDARasterBuffer.h"

#include "Utility/CUDA/CudaHelper.cuh"

#include "Raster Engine\RasterBackend.h"
#include "Raster Engine\OpenGL\OpenGLBuffer.h"
#include <cuda_gl_interop.h>

CUDARasterBuffer::CUDARasterBuffer(CUDADevice* device,uint size) {

	cudaSetDevice(device->order);

	Buffer* rasterBuffer = RasterBackend::CreateBuffer(size);

	CUDARasterBuffer* buffer;

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

CUDARasterBuffer::~CUDARasterBuffer() {


}
