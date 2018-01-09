#pragma once

#include "glm/glm.hpp"

class GPUExecutePolicy {

public:

	GPUExecutePolicy();
	GPUExecutePolicy(glm::uvec3, glm::uvec3, int, int);
	~GPUExecutePolicy();

	glm::uvec3 gridsize;
	glm::uvec3 blocksize;
	int sharedMemory;
	int stream;

private:

	

};