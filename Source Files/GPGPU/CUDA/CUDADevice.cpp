#include "CUDADevice.h"

CUDADevice::CUDADevice(uint o) :
GPUDevice(o)
{

	api = CUDA;

}

CUDADevice::~CUDADevice() {


}
