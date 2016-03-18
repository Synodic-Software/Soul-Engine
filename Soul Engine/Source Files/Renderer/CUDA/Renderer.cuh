#pragma once

#include "Utility\CUDAIncludes.h"
#include "Ray Engine/CUDA/RayJob.cuh"

__host__ void Integrate(RayJob*, const uint);
