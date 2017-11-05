#include "GPUExecutePolicy.h"

GPUExecutePolicy::GPUExecutePolicy() {
	
}

GPUExecutePolicy::GPUExecutePolicy(glm::vec3 grid, glm::vec3 block, int shared, int str) :
	gridsize(grid), blocksize(block), sharedMemory(shared), stream(str) {

}

GPUExecutePolicy::~GPUExecutePolicy() {

}