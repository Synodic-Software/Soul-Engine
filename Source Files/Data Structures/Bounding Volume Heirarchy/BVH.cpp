#include "BVH.h"
#include "CUDA/BVH.cuh"
#include "Compute/GPUManager.h"

BVH::BVH():
bvh(GPUManager::GetBestGPU())
{

}

void BVH::Build(int size, ComputeBuffer<BVHData>& data, ComputeBuffer<uint64>& mortonCodes, ComputeBuffer<Face>& faces, ComputeBuffer<Vertex>& vertices) {

	if (size > 0) {

		const int nodeSize = size * 2 - 1;

		bvh.ResizeDevice(nodeSize);

		data[0].currentSize = size;
		data[0].bvh = bvh.DataDevice();
		data.TransferToDevice();

		GPUDevice device = GPUManager::GetBestGPU();

		const uint blockSize = 64;
		const GPUExecutePolicy normalPolicy(glm::vec3((size + blockSize - 1) / blockSize, 1, 1), glm::vec3(blockSize, 1, 1), 0, 0);


		device.Launch(normalPolicy, Reset, 
			size, 
			bvh.DataDevice(), 
			faces.DataDevice(), 
			vertices.DataDevice(), 
			mortonCodes.DataDevice(), 
			size - 1);

		device.Launch(normalPolicy, BuildTree, 
			size, 
			data.DataDevice(), 
			bvh.DataDevice(), 
			mortonCodes.DataDevice(), 
			size - 1);

	}

}