#pragma once

#include "Ray Engine/CUDA/RayJob.cuh"
#include "Engine Core\Scene\CUDA/Scene.cuh"
#include "GPGPU/GPUBuffer.h"

__host__ void ProcessJobs(GPUBuffer<RayJob>&, const Scene*);
__host__ void GPUInitialize();
__host__ void GPUTerminate();