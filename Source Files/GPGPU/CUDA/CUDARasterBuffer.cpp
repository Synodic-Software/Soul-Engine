#include "CUDARasterBuffer.h"

#include "Utility/CUDA/CudaHelper.cuh"

#include "Raster Engine\RasterBackend.h"
#include <cuda_gl_interop.h>
#include "Multithreading\Scheduler.h"

/*
 *    Constructor.
 *    @param [in,out]	device	If non-null, the device.
 *    @param 		 	size  	The size.
 */

CUDARasterBuffer::CUDARasterBuffer(CUDADevice* device, uint size) {

	cudaSetDevice(device->order);

	rasterBuffer = RasterBackend::CreateBuffer(size);

	if (RasterBackend::backend == OpenGL) {

		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [this]() {
			RasterBackend::MakeContextCurrent();

			OpenGLBuffer* oglBuffer = static_cast<OpenGLBuffer*>(rasterBuffer);


			CudaCheck(cudaGraphicsGLRegisterBuffer(&cudaBuffer
				, oglBuffer->GetBufferID()
				, cudaGraphicsRegisterFlagsWriteDiscard));

		});
		Scheduler::Block();

	}
	else {
		//TODO
		//Vulkan Stuff


	}

}

/* Destructor. */
CUDARasterBuffer::~CUDARasterBuffer() {


}

/* Map resources. */
void CUDARasterBuffer::MapResources() {

	if (RasterBackend::backend == OpenGL) {

		CUDARasterBuffer* buff = this;
		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [&buff]() {
			RasterBackend::MakeContextCurrent();

			CudaCheck(cudaGraphicsMapResources(1, &buff->cudaBuffer, 0));

			size_t num_bytes;
			CudaCheck(cudaGraphicsResourceGetMappedPointer((void **)&buff->data, &num_bytes,
				buff->cudaBuffer));

		});


		Scheduler::Block();

	}
	else {
		//TODO
		//Vulkan Stuff


	}
}

/* Unmap resources. */
void CUDARasterBuffer::UnmapResources() {

	if (RasterBackend::backend == OpenGL) {

		CUDARasterBuffer* buff = this;

		Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [&buff]() {
			RasterBackend::MakeContextCurrent();
			CudaCheck(cudaGraphicsUnmapResources(1, &buff->cudaBuffer, 0));
		});
		Scheduler::Block();
	}
	else {
		//TODO
		//Vulkan Stuff


	}
}

/*
 *    Bind data.
 *    @param	pos	The position.
 */

void CUDARasterBuffer::BindData(uint pos) {


	if (RasterBackend::backend == OpenGL) {

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