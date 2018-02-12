#pragma once

#include "glm/glm.hpp"
#include "Metrics.h"

class GPUExecutePolicy {

public:

	//Constructors + Destructors
	GPUExecutePolicy() = default;
	GPUExecutePolicy(uint, uint, int, int);
	GPUExecutePolicy(glm::uvec3, glm::uvec3, int, int);


	//Policy Helpers
	uint GetThreadCount() const;


	glm::uvec3 gridsize;
	glm::uvec3 blocksize;
	int sharedMemory;
	int stream;

};