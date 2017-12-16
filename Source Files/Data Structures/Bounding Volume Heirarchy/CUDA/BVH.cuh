#pragma once

#include "Utility\CUDA\CUDAHelper.cuh"
#include "Data Structures/Bounding Volume Heirarchy/BVH.h"

__global__ void BuildTree(const uint n, BVHData* data, Node* nodes, uint64* mortonCodes, const uint leafOffset);
__global__ void Reset(const uint n, Node* nodes, Face* faces, Vertex* vertices, uint64* mortonCodes, const uint leafOffset);
