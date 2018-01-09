#include "BVH.h"
#include "CUDA/BVH.cuh"
#include "GPGPU/GPUManager.h"

BVH::BVH() {

	bvh.Move(GPUManager::GetBestGPU());

}

BVH::~BVH() {

}

void BVH::Build(int size, GPUBuffer<BVHData>& data, GPUBuffer<uint64>& mortonCodes, GPUBuffer<Face>& faces, GPUBuffer<Vertex>& vertices) {

	if (size > 0) {

		const int nodeSize = size * 2 - 1;

		if (nodeSize > bvh.DeviceCapacity()) {
			bvh.resize(nodeSize);
		}

		data[0].currentSize = size;
		data[0].bvh = bvh.DeviceData();
		data.TransferToDevice();

		GPUDevice device = GPUManager::GetBestGPU();

		const uint blockSize = 64;
		const GPUExecutePolicy normalPolicy(glm::vec3((size + blockSize - 1) / blockSize, 1, 1), glm::vec3(blockSize, 1, 1), 0, 0);

		auto bvhP = bvh.DeviceData();
		auto faceP = faces.DeviceData();
		auto vertP = vertices.DeviceData();
		auto mortP = mortonCodes.DeviceData();
		auto leafSize = size - 1;
		auto dataP = data.DeviceData();

		device.Launch(normalPolicy, Reset, size, bvhP, faceP, vertP, mortP, leafSize);
		device.Launch(normalPolicy, BuildTree, size, dataP, bvhP, mortP, leafSize);

	}

}