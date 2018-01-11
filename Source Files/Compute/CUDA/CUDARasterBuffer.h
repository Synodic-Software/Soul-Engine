#pragma once

#include "Multithreading\Scheduler.h"
#include "Compute\DeviceRasterBuffer.h"
#include "Raster Engine\RasterBackend.h"
#include "Raster Engine\OpenGL\OpenGLBuffer.h"

#include "Metrics.h"

#include "Compute\GPUDevice.h"
#include "Utility/CUDA/CudaHelper.cuh"

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>


template <class T>
class CUDARasterBuffer :public DeviceRasterBuffer<T> {

public:

	/*
	 *    Constructor.
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 *    @param 		 	parameter2	The second parameter.
	 */

	CUDARasterBuffer(GPUDevice& _device, uint _count)
		: DeviceRasterBuffer(_device, _count) {

		CUDARasterBuffer::resize(_count);

		device = _device.GetOrder();
	}
	/* Destructor. */
	~CUDARasterBuffer() {}

	/* Map resources. */
	void MapResources() override {

		if (RasterBackend::backend == OpenGL) {

			CUDARasterBuffer* buff = this;
			Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [&buff]() {
				RasterBackend::MakeContextCurrent();

				CudaCheck(cudaGraphicsMapResources(1, &buff->cudaBuffer, 0));

				size_t num_bytes;
				CudaCheck(cudaGraphicsResourceGetMappedPointer((void **)&buff->DataDevice, &num_bytes,
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
	void UnmapResources() override {

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

	void BindData(uint pos) override {


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

	void Resize(uint newSize) override {

		DeviceBuffer<T>::resize(newSize);

		if (newSize > 0) {

			CudaCheck(cudaSetDevice(device));
			rasterBuffer = RasterBackend::CreateBuffer(newSize*sizeof(T));

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
	}

private:

	Buffer* rasterBuffer = nullptr;

	struct cudaGraphicsResource* cudaBuffer;

	int device;

};