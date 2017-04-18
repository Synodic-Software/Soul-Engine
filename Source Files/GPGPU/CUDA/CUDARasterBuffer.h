#pragma once
#include "GPGPU\GPURasterBuffer.h"
#include "Metrics.h"

#include "GPGPU\CUDA\CUDADevice.h"
#include "Raster Engine\OpenGL\OpenGLBuffer.h"

class CUDARasterBuffer :public GPURasterBuffer {

public:
	CUDARasterBuffer(CUDADevice*, uint);
	~CUDARasterBuffer();

	void MapResources();
	void UnmapResources();
	void BindData(uint);

protected:

private:
	Buffer* rasterBuffer = nullptr;
	struct cudaGraphicsResource* cudaBuffer;

};