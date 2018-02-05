#include "BVH.h"
#include "CUDA/BVH.cuh"
#include "Compute/ComputeManager.h"

BVH::BVH() :
	innerNodes(S_BEST_GPU),
	leafNodes(S_BEST_GPU)
{

}

void BVH::Build(int size, ComputeBuffer<BVHData>& data, ComputeBuffer<uint64>& mortonCodes, ComputeBuffer<BoundingBox>& boxes) {

	if (size > 0) {

		const GPUExecutePolicy normalPolicy(size, 64, 0, 0);

		innerNodes.ResizeDevice(size); // size-1 nodes are actually used. 1 is tacked on to remove conditional statements
		leafNodes.ResizeDevice(size);

		data[0].leafSize = size;
		data[0].leafNodes = leafNodes.DataDevice();
		data[0].innerNodes = innerNodes.DataDevice();
		data.TransferToDevice();

		ComputeDevice device = S_BEST_GPU;


		device.Launch(normalPolicy, Reset,
			size,
			innerNodes.DataDevice(),
			leafNodes.DataDevice());

		
		/*innerNodes.TransferToHost();
		leafNodes.TransferToHost();

		for (auto id = 0; id < size - 1; ++id) {
			const auto node = innerNodes[id];
			std::cout << "Node" << " " << id << ": " << node.childLeft << " " << node.childRight << "        " << node.atomic << " " << node.rangeLeft << " " << node.rangeRight << std::endl;
		}*/

		//std::cout << std::endl;


		device.Launch(normalPolicy, BuildTree,
			size,
			data.DataDevice(),
			innerNodes.DataDevice(),
			leafNodes.DataDevice(),
			mortonCodes.DataDevice(),
			boxes.DataDevice());

		/*innerNodes.TransferToHost();
		leafNodes.TransferToHost();

		for (auto id = 0; id < size - 1; ++id) {
			const auto node = innerNodes[id];
			std::cout << "Node" << " " << id << ": " << node.childLeft << " " << node.childRight << "        " << node.atomic << " " << node.rangeLeft << " " << node.rangeRight << std::endl;
		}*/
	}

}