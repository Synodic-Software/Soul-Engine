#pragma once

#include "GPUBuffer.h"
#include "GPGPU\CUDA\CUDADevice.h"

/* Buffer for GPU raster. */
class GPURasterBuffer :public GPUBuffer{

public:
	/* Default constructor. */
	GPURasterBuffer();
	/* Destructor. */
	~GPURasterBuffer();

	/* Map resources. */
	virtual void MapResources()=0;
	/* Unmap resources. */
	virtual void UnmapResources()=0;

	/*
	 *    Bind data.
	 *    @param	parameter1	The first parameter.
	 */

	virtual void BindData(uint) = 0;

protected:

private:

};