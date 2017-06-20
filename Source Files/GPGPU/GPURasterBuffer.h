#pragma once

#include "GPUBuffer.h"
#include "GPGPU\CUDA\CUDADevice.h"

class GPURasterBuffer :public GPUBuffer{

public:
	GPURasterBuffer();
	~GPURasterBuffer();

	virtual void MapResources()=0;
	virtual void UnmapResources()=0;
	virtual void BindData(uint) = 0;

protected:

private:

};