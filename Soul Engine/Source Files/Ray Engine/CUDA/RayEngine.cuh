#pragma once

#include "Engine Core/BasicDependencies.h"
#include "Ray Engine/RayJob.cuh"
#include "Engine Core\Scene\Scene.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__host__ void ProcessJobs(std::vector<RayJob*>&, const Scene*);