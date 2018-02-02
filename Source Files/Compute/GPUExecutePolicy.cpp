#include "GPUExecutePolicy.h"

GPUExecutePolicy::GPUExecutePolicy()
{

}

GPUExecutePolicy::GPUExecutePolicy(uint size_, uint blockSize_, int sharedBytes, int stream_) :
	gridsize((size_ + blockSize_ - 1) / blockSize_, 1, 1), 
	blocksize(blockSize_, 1, 1),
	sharedMemory(sharedBytes), 
	stream(stream_)
{

}

GPUExecutePolicy::GPUExecutePolicy(glm::uvec3 grid, glm::uvec3 block, int sharedBytes, int stream_) :
	gridsize(grid), 
	blocksize(block), 
	sharedMemory(sharedBytes), 
	stream(stream_) 
{

}