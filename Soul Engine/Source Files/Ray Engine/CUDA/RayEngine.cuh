#pragma once

#include "Engine Core/BasicDependencies.h"
#include "Ray Engine/RayJob.cuh"
#include "Engine Core\Scene\Scene.cuh"

__host__ void ProcessJobs(RayJob*, const Scene*);