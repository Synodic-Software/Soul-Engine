#pragma once

#include "Utility\CUDAIncludes.h"
#include "Ray Engine/CUDA/RayJob.cuh"
#include "Engine Core\Scene\CUDA/Scene.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>


__host__ void ProcessJobs(std::vector<RayJob*>&, const Scene*);
__host__ void ClearResults(std::vector<RayJob*>&);
__host__ void Cleanup();
