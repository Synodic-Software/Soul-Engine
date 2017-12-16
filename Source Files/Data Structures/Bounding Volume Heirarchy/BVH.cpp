#include "BVH.h"

#include "Utility/Logger.h"

BVH::BVH() {

	allocatedSize = 0;
	bvhDataHost.currentSize = 0;
	bvh = nullptr;

}

BVH::~BVH() {

	if (bvh) {
		CudaCheck(cudaFree(bvh));
	}

}

void BVH::Build(uint size, BVHData*& data, uint64* mortonCodes, Face * faces, Vertex * vertices) {

	if (size > 0) {
		if (size > allocatedSize) {

			allocatedSize = glm::max(uint(allocatedSize * 1.5f), (size * 2) - 1);

			if (bvh) {
				CudaCheck(cudaFree(bvh));
			}

			CudaCheck(cudaMalloc((void**)&bvh, allocatedSize * sizeof(Node)));
			bvhDataHost.bvh = bvh;
		}

		bvhDataHost.currentSize = size;
		CudaCheck(cudaMemcpy(data, &bvhDataHost, sizeof(BVHData), cudaMemcpyHostToDevice));

		uint blockSize = 64;
		uint gridSize = (size + blockSize - 1) / blockSize;

		CudaCheck(cudaDeviceSynchronize());

		Reset << <gridSize, blockSize >> > (size, bvh, faces, vertices, mortonCodes, size - 1);
		CudaCheck(cudaPeekAtLastError());
		CudaCheck(cudaDeviceSynchronize());

		BuildTree << <gridSize, blockSize >> > (size, data, bvh, mortonCodes, size - 1);
		CudaCheck(cudaPeekAtLastError());
		CudaCheck(cudaDeviceSynchronize());
	}

}