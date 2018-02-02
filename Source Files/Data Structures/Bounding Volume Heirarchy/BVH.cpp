#include "BVH.h"
#include "CUDA/BVH.cuh"
#include "Compute/ComputeManager.h"

BVH::BVH() :
	bvh(S_BEST_GPU)
{

}

void BVH::Build(int size, ComputeBuffer<BVHData>& data, ComputeBuffer<uint64>& mortonCodes, ComputeBuffer<Face>& faces, ComputeBuffer<Vertex>& vertices) {

	if (size > 0) {

		const int nodeSize = size * 2 - 1;

		bvh.ResizeDevice(nodeSize);

		data[0].currentSize = size;
		data[0].bvh = bvh.DataDevice();
		data.TransferToDevice();

		ComputeDevice device = S_BEST_GPU;
		ComputeDevice cpu = S_BEST_CPU;

		const GPUExecutePolicy normalPolicy(size, 64, 0, 0);

		cpu.Launch(normalPolicy, Reset,
			size,
			size - 1,
			bvh.DataDevice(),
			faces.DataDevice(),
			vertices.DataDevice());

		device.Launch(normalPolicy, BuildTree,
			size,
			size - 1,
			data.DataDevice(),
			bvh.DataDevice(),
			mortonCodes.DataDevice());

	}

}