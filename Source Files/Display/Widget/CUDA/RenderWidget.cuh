#pragma once

#include "Ray Engine/CUDA/RayJob.cuh"
#include "Utility\Includes\GLMIncludes.h"

__host__ void Integrate(RayJob*, glm::vec4*, glm::vec4*, const uint);