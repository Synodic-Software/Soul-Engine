#pragma once

#include "Ray Engine/CUDA/RayJob.cuh"
#include "glm/glm.hpp"

__host__ void Integrate(uint, glm::vec4*, glm::vec4*, int*,const uint);