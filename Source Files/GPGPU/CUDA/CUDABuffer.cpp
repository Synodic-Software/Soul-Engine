#include "CUDABuffer.h"

#include "Utility/CUDA/CudaHelper.cuh"

CUDABuffer::CUDABuffer() {

}

CUDABuffer::CUDABuffer(CUDADevice* deviceIn, uint sizeIn) {

	//CudaCheck(cudaMalloc((void**)&bufferData, sizeIn));

}

CUDABuffer::~CUDABuffer() {


}
