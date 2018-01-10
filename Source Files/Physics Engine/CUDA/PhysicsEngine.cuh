#pragma once

#include "GPGPU/ComputeBuffer.h"
#include "Engine Core/Scene/Scene.h"


__host__ void ProcessScene(ComputeBuffer<BVHData>&);
