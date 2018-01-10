#include "GPUExecutePolicy.h"

GPUExecutePolicy::GPUExecutePolicy() {
	
}

GPUExecutePolicy::GPUExecutePolicy(glm::uvec3 grid, glm::uvec3 block, int shared, int str) :
	gridsize(grid), blocksize(block), sharedMemory(shared), stream(str) {

}

GPUExecutePolicy::~GPUExecutePolicy() {

}