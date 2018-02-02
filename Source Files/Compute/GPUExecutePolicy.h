#pragma once

#include "glm/glm.hpp"
#include "Metrics.h"

class GPUExecutePolicy {

public:

	GPUExecutePolicy();
	GPUExecutePolicy(uint, uint, int, int);
	GPUExecutePolicy(glm::uvec3, glm::uvec3, int, int);
	~GPUExecutePolicy() = default;

	glm::uvec3 gridsize;
	glm::uvec3 blocksize;
	int sharedMemory;
	int stream;

private:

	

};