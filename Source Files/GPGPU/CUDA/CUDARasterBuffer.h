#pragma once

#include "Multithreading\Scheduler.h"
#include "GPGPU\GPURasterBuffer.h"
#include "Raster Engine\RasterBackend.h"
#include "Raster Engine\OpenGL\OpenGLBuffer.h"

#include "Metrics.h"

#include "GPGPU\CUDA\CUDADevice.h"
#include "Utility/CUDA/CudaHelper.cuh"

#include <cuda_gl_interop.h>

/* Buffer for cuda raster. */
template <class T>
class CUDARasterBuffer :public GPURasterBuffer<T> {

public:

	/*
	 *    Constructor.
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 *    @param 		 	parameter2	The second parameter.
	 */

	CUDARasterBuffer(GPUDevice& _device, uint _byteCount)
		: GPURasterBuffer(_device, _byteCount) {

		cudaSetDevice(_device.order);

		rasterBuffer = RasterBackend::CreateBuffer(_byteCount);

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
	~CUDARasterBuffer() {}

	/* Map resources. */
	void MapResources() {

		if (RasterBackend::backend == OpenGL) {

			CUDARasterBuffer* buff = this;
			Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [&buff]() {
				RasterBackend::MakeContextCurrent();

				CudaCheck(cudaGraphicsMapResources(1, &buff->cudaBuffer, 0));

				size_t num_bytes;
				CudaCheck(cudaGraphicsResourceGetMappedPointer((void **)&buff->deviceData, &num_bytes,
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
	void UnmapResources() {

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
	 *    @param	parameter1	The first parameter.
	 */

	void BindData(uint pos) {


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

protected:

private:
	/* Buffer for raster data */
	Buffer* rasterBuffer = nullptr;
	/* A cuda graphics resource*. */
	struct cudaGraphicsResource* cudaBuffer;

};