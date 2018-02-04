#include "BVH.h"
#include "CUDA/BVH.cuh"
#include "Compute/ComputeManager.h"

BVH::BVH() :
	innerNodes(S_BEST_GPU),
	leafNodes(S_BEST_GPU)
{

}

void BVH::Build(int size, ComputeBuffer<BVHData>& data, ComputeBuffer<uint64>& mortonCodes, ComputeBuffer<Face>& faces, ComputeBuffer<Vertex>& vertices) {

	if (size > 0) {

		innerNodes.ResizeDevice(size - 1);
		leafNodes.ResizeDevice(size);

		data[0].leafSize = size;
		data[0].leafNodes = leafNodes.DataDevice();
		data[0].innerNodes = innerNodes.DataDevice();
		data.TransferToDevice();

		ComputeDevice device = S_BEST_GPU;

		const GPUExecutePolicy normalPolicy(size, 64, 0, 0);

		device.Launch(normalPolicy, Reset,
			size,
			size - 1,
			innerNodes.DataDevice(),
			leafNodes.DataDevice(),
			faces.DataDevice(),
			vertices.DataDevice());

		
		innerNodes.TransferToHost();
		leafNodes.TransferToHost();

		auto id = 0;
		for (auto& node : innerNodes) {

			std::cout << "Node" << " " << id << ": " << node.childLeft << " " << node.childLeft << "        " << node.atomic << " " << node.rangeLeft << " " << node.rangeRight << std::endl;

			id++;
		}

		std::cout << std::endl;


		device.Launch(normalPolicy, BuildTree,
			size,
			size - 1,
			data.DataDevice(),
			innerNodes.DataDevice(),
			leafNodes.DataDevice(),
			mortonCodes.DataDevice());

		innerNodes.TransferToHost();
		leafNodes.TransferToHost();

		id = 0;
		for (auto& node : innerNodes) {

			std::cout << "Node" << " " << id << ": " << node.childLeft << " " << node.childLeft << "        " << node.atomic << " " << node.rangeLeft << " " << node.rangeRight << std::endl;

			id++;
		}
	}

}