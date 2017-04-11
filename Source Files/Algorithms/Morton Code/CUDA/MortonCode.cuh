#pragma once

#include "Metrics.h"
#include "Bounding Volume Heirarchy\BVH.h"

#include "Engine Core\Object\Face.h"
#include "Engine Core\Object\Vertex.h"

namespace MortonCode{

	__global__ void Compute(const uint n, uint64* mortonCodes, Face* faces, Vertex* vertices,const BoundingBox box);

}