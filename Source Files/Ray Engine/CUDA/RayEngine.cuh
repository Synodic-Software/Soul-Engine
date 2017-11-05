#pragma once

#include "Ray Engine/CUDA/RayJob.cuh"
#include "Engine Core\Scene\CUDA/Scene.cuh"
#include "GPGPU/GPUBuffer.h"

__global__ void ExecuteJobs(const uint n, Ray* rays, const Scene* scene, int* counter);
__global__ void ProcessHits(const uint n, RayJob* job, int jobSize, Ray* rays, Ray* raysNew, const Scene* scene, int * nAtomic, curandState* randomState);
__global__ void EngineSetup(const uint n, RayJob* jobs, int jobSize);
__global__ void RaySetup(const uint n, int jobSize, RayJob* job, Ray* rays, int* nAtomic, curandState* randomState);
__global__ void RandomSetup(const uint n, curandState* randomState, const uint raySeed);