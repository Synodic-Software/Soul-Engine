#include "MortonCode.cuh"
#include "Utility\CUDA\CUDAHelper.cuh"
#include "Utility\Logger.h"
#include "Utility\Includes\GLMIncludes.h"
#include <inttypes.h>
#include <stdint.h>


__inline__ __host__ __device__ uint64 Split64(unsigned int a) {
	uint64 x = a;
	x = (x | x << 32) & 0x1f00000000ffff;
	x = (x | x << 16) & 0x1f0000ff0000ff;
	x = (x | x << 8) & 0x100f00f00f00f00f;
	x = (x | x << 4) & 0x10c30c30c30c30c3;
	x = (x | x << 2) & 0x1249249249249249;
	return x;
}

__inline__ __host__ __device__ uint Unsplit64(const uint64 m) {
	uint64 x = m & 0x1249249249249249;
	x = (x ^ (x >> 2)) & 0x10c30c30c30c30c3;
	x = (x ^ (x >> 4)) & 0x100f00f00f00f00f;
	x = (x ^ (x >> 8)) & 0x1f0000ff0000ff;
	x = (x ^ (x >> 16)) & 0x1f00000000ffff;
	x = (x ^ (x >> 32)) & 0x1fffff;
	return x;
}

__host__ __device__ uint64 MortonCode::Calculate64(const glm::vec3& data) {

	uint max = 2097151;
	uint x = uint(data.x*max);
	uint y = uint(data.y*max);
	uint z = uint(data.z*max);

	uint64 answer = 0;
	answer |= Split64(x) | Split64(y) << 1 | Split64(z) << 2;
	return answer;
}

__host__ __device__ glm::vec3 MortonCode::Decode64(uint64 m) {

	uint x = Unsplit64(m);
	uint y = Unsplit64(m >> 1);
	uint z = Unsplit64(m >> 2);

	float max = 2097151;
	glm::vec3 data = glm::vec3(x / max, y / max, z / max);

	return data;
}


__global__ void MortonCode::ComputeGPU(const uint n, uint64* mortonCodes, Face* faces, Vertex* vertices) {

	uint index = getGlobalIdx_1D_1D();

	if (index >= n) {
		return;
	}

	glm::uvec3 ind = faces[index].indices;
	glm::vec3 centroid = ((vertices + ind.x)->position + (vertices + ind.y)->position + (vertices + ind.z)->position) / 3.0f;

	mortonCodes[index] = Calculate64(centroid);
}