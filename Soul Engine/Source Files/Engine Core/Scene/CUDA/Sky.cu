#include "Sky.cuh"

Sky::Sky(std::string texName) {

	image = new Image();

	CudaCheck(cudaDeviceSynchronize());
	image->LoadFromFile(texName.c_str(),true,false);
	CudaCheck(cudaDeviceSynchronize());
}

__device__ glm::vec3 Sky::ExtractColour(const glm::vec3& direction){
	float4 col = tex2D<float4>(image->texObj, 0.5f + atan2f(direction.z, direction.x) / (2 * PI), 0.5f - asinf(direction.y)/PI);

	return glm::vec3(col.x, col.y, col.z);
}
