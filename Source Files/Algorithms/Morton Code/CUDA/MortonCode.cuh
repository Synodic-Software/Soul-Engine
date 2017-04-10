#pragma once

#include "Metrics.h"
#include "Bounding Volume Heirarchy\BVH.h"

namespace MortonCode{

	__global__ void Compute(const uint n, uint64* mortonCodes, Face** faceList, Object** objectList, BoundingBox* box);

}