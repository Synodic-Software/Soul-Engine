#pragma once

#include "Ray Engine/CUDA/RayJob.cuh"

__host__ void Integrate(RayJob*,glm::vec4*, glm::vec4*, const uint);
