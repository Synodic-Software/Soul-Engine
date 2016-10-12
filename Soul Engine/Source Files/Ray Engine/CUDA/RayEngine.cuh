#pragma once

#include "Ray Engine/CUDA/RayJob.cuh"
#include "Engine Core\Scene\CUDA/Scene.cuh"

__host__ void ProcessJobs(std::vector<RayJob*>&, const Scene*);
__host__ void ClearResults(std::vector<RayJob*>&);
__host__ void Cleanup();
