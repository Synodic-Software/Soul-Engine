#pragma once

#include "GPGPU/GPUBuffer.h"
#include "Engine Core/Scene/Scene.h"


__host__ void ProcessScene(GPUBuffer<BVHData>&);
