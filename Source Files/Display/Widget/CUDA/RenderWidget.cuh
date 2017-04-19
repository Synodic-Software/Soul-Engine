#pragma once

#include "Ray Engine/CUDA/RayJob.cuh"
#include "Utility\Includes\GLMIncludes.h"

__host__ void Integrate(uint, glm::vec4*, glm::vec4*, const uint);