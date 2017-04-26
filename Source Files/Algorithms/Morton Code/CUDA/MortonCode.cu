#include "MortonCode.cuh"
#include "Utility\CUDA\CUDAHelper.cuh"
#include "Utility\Logger.h"
#include "Utility\Includes\GLMIncludes.h"
#include <inttypes.h>
#include <stdint.h>

//morton codes precomputed

__inline__ __host__ __device__ uint64 Split(unsigned int a) {
	uint64 x = a;
	x = (x | x << 32) & 0x1f00000000ffff;  
	x = (x | x << 16) & 0x1f0000ff0000ff; 
	x = (x | x << 8) & 0x100f00f00f00f00f;
	x = (x | x << 4) & 0x10c30c30c30c30c3;
	x = (x | x << 2) & 0x1249249249249249;
	return x;
}

__host__ __device__ uint64 MortonCode::CalculateMorton(const glm::vec3& data) {

	uint max= 2097151;
	uint x = uint(data.x*max);
	uint y = uint(data.y*max);
	uint z = uint(data.z*max);

	uint64 answer = 0;
	answer |= Split(x) | Split(y) << 1 | Split(z) << 2;
	return answer;
}


__global__ void MortonCode::Compute(const uint n, uint64* mortonCodes, Face* faces, Vertex* vertices) {

	uint index = getGlobalIdx_1D_1D();

	if (index >= n) {
		return;
	}

	glm::uvec3 ind = faces[index].indices;
	glm::vec3 centroid = ((vertices + ind.x)->position + (vertices + ind.y)->position + (vertices + ind.z)->position) / 3.0f;

	mortonCodes[index] = CalculateMorton(centroid);
}