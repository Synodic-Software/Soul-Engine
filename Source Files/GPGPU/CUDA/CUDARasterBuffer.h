//---------------------------------------------------------------------------------------------------
//@file	N:\Documents\Soul Engine\Source Files\GPGPU\CUDA\CUDARasterBuffer.h.
//Declares the cuda raster buffer class.

#pragma once
#include "GPGPU\GPURasterBuffer.h"
#include "Metrics.h"

#include "GPGPU\CUDA\CUDADevice.h"
#include "Raster Engine\OpenGL\OpenGLBuffer.h"

//Buffer for cuda raster.
class CUDARasterBuffer :public GPURasterBuffer {

public:

	//---------------------------------------------------------------------------------------------------
	//Constructor.
	//@param [in,out]	parameter1	If non-null, the first parameter.
	//@param 		 	parameter2	The second parameter.

	CUDARasterBuffer(CUDADevice*, uint);
	//Destructor.
	~CUDARasterBuffer();

	//Map resources.
	void MapResources();
	//Unmap resources.
	void UnmapResources();

	//---------------------------------------------------------------------------------------------------
	//Bind data.
	//@param	parameter1	The first parameter.

	void BindData(uint);

protected:

private:
	//Buffer for raster data
	Buffer* rasterBuffer = nullptr;
	//A cuda graphics resource*.
	struct cudaGraphicsResource* cudaBuffer;

};