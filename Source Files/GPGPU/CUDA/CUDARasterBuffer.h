#pragma once
#include "GPGPU\GPURasterBuffer.h"
#include "Metrics.h"

class CUDARasterBuffer :public GPURasterBuffer {

public:
	CUDARasterBuffer(uint);
	~CUDARasterBuffer();

protected:

private:

};