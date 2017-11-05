#pragma once

#include "glm/glm.hpp"

class GPUExecutePolicy {

public:

	GPUExecutePolicy();
	GPUExecutePolicy(glm::vec3, glm::vec3, int, int);
	~GPUExecutePolicy();

	glm::vec3 gridsize;
	glm::vec3 blocksize;
	int sharedMemory;
	int stream;

private:

	

};