#include "BVH.h"
#include "CUDA/BVH.cuh"
#include "Compute/GPUManager.h"

BVH::BVH() {

	bvh.Move(GPUManager::GetBestGPU());

}

BVH::~BVH() {

}

void BVH::Build(int size, ComputeBuffer<BVHData>& data, ComputeBuffer<uint64>& mortonCodes, ComputeBuffer<Face>& faces, ComputeBuffer<Vertex>& vertices) {

	if (size > 0) {

		const int nodeSize = size * 2 - 1;

		bvh.Resize(nodeSize);

		data[0].currentSize = size;
		data[0].bvh = bvh.DataDevice();
		data.TransferToDevice();

		GPUDevice device = GPUManager::GetBestGPU();

		const uint blockSize = 64;
		const GPUExecutePolicy normalPolicy(glm::vec3((size + blockSize - 1) / blockSize, 1, 1), glm::vec3(blockSize, 1, 1), 0, 0);

		auto bvhP = bvh.DataDevice();
		auto faceP = faces.DataDevice();
		auto vertP = vertices.DataDevice();
		auto mortP = mortonCodes.DataDevice();
		auto leafSize = size - 1;
		auto dataP = data.DataDevice();

		device.Launch(normalPolicy, Reset, size, bvhP, faceP, vertP, mortP, leafSize);
		device.Launch(normalPolicy, BuildTree, size, dataP, bvhP, mortP, leafSize);

	}

}