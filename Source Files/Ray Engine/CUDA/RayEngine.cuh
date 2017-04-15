#pragma once

#include "Ray Engine/CUDA/RayJob.cuh"
#include "Engine Core\Scene\CUDA/Scene.cuh"
#include <vector>

__host__ void ProcessJobs(std::vector<RayJob>&, const Scene*);
__host__ void GPUInitialize();
__host__ void GPUTerminate();