#include "CUDARasterBuffer.h"

#include "Utility/CUDA/CudaHelper.cuh"

#include "Raster Engine\RasterBackend.h"
#include <cuda_gl_interop.h>
#include "Multithreading\Scheduler.h"

CUDARasterBuffer::CUDARasterBuffer(CUDADevice* device, uint size) {

	cudaSetDevice(device->order);

	rasterBuffer = RasterBackend::CreateBuffer(size);

	if (RasterBackend::backend == OpenGL) {

		OpenGLBuffer* oglBuffer = static_cast<OpenGLBuffer*>(rasterBuffer);

		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [this,&oglBuffer]() {
			RasterBackend::MakeContextCurrent();

			CudaCheck(cudaGraphicsGLRegisterBuffer(&cudaBuffer
				, oglBuffer->GetBufferID()
				, cudaGraphicsRegisterFlagsWriteDiscard));


			CudaCheck(cudaGraphicsMapResources(1, &cudaBuffer, 0));
			size_t num_bytes;
			CudaCheck(cudaGraphicsResourceGetMappedPointer((void **)&bufferData, &num_bytes,
				cudaBuffer));

		});
		Scheduler::Block();
	}
	else {
		//TODO
		//Vulkan Stuff


	}

}

CUDARasterBuffer::~CUDARasterBuffer() {


}

void CUDARasterBuffer::MapResources() {

	if (RasterBackend::backend == OpenGL) {

		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [this]() {
			RasterBackend::MakeContextCurrent();

			CudaCheck(cudaGraphicsMapResources(1, &cudaBuffer, 0));

			size_t num_bytes;
			CudaCheck(cudaGraphicsResourceGetMappedPointer((void **)&bufferData, &num_bytes,
				cudaBuffer));

		});
		Scheduler::Block();

	}
	else {
		//TODO
		//Vulkan Stuff


	}
}

void CUDARasterBuffer::UnmapResources() {

	if (RasterBackend::backend == OpenGL) {
		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [this]() {

			CudaCheck(cudaGraphicsUnmapResources(1, &cudaBuffer, 0));

		});
		Scheduler::Block();

	}
	else {
		//TODO
		//Vulkan Stuff


	}
}

void CUDARasterBuffer::BindData(uint pos) {


	if (RasterBackend::backend == OpenGL) {

		OpenGLBuffer* oglBuffer = static_cast<OpenGLBuffer*>(rasterBuffer);

		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [this, &pos]() {
			RasterBackend::MakeContextCurrent();
			OpenGLBuffer* oglBuffer = static_cast<OpenGLBuffer*>(rasterBuffer);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, pos, oglBuffer->GetBufferID());


		});
		Scheduler::Block();

	}
	else {
		//TODO
		//Vulkan Stuff


	}


}