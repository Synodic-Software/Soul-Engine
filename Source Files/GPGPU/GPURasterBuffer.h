#pragma once

#include "GPUBuffer.h"
#include "GPGPU\CUDA\CUDADevice.h"

/* Buffer for GPU raster. */
template <class T>
class GPURasterBuffer :public GPUBuffer<T>{

public:
	/* Default constructor. */
	GPURasterBuffer(GPUDevice& _device, uint _byteCount)
		: GPUBuffer(_device, _byteCount) {
		
	}
	/* Destructor. */
	virtual ~GPURasterBuffer(){}

	/* Map resources. */
	virtual void MapResources() = 0;
	/* Unmap resources. */
	virtual void UnmapResources() = 0;

	/*
	 *    Bind data.
	 *    @param	parameter1	The first parameter.
	 */

	virtual void BindData(uint) = 0;

protected:

private:

};