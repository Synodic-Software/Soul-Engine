#include "BVH.h"
#include "CUDA/BVH.cuh"
#include "Compute/ComputeManager.h"

BVH::BVH() :
	nodes(S_BEST_GPU)
{

}

void BVH::Build(int size, ComputeBuffer<BVHData>& data, ComputeBuffer<uint64>& mortonCodes, ComputeBuffer<BoundingBox>& boxes) {

	if (size > 0) {

		const GPUExecutePolicy normalPolicy(size, 64, 0, 0);

		nodes.ResizeDevice(size); // size-1 nodes are actually used. 1 is tacked on to remove conditional statements

		data[0].leafSize = size;
		data[0].nodes = nodes.DataDevice();
		data.TransferToDevice();

		ComputeDevice device = S_BEST_GPU;


		device.Launch(normalPolicy, Reset,
			size,
			nodes.DataDevice());

		
		/*nodes.TransferToHost();

		for (auto id = 0; id < size - 1; ++id) {
			const auto node = nodes[id];
			std::cout << "Node" << " " << id << ": " << node.childLeft << " " << node.childRight << "        " << node.atomic << " " << node.rangeLeft << " " << node.rangeRight << std::endl;
		}*/

		//std::cout << std::endl;


		device.Launch(normalPolicy, BuildTree,
			size,
			data.DataDevice(),
			nodes.DataDevice(),
			mortonCodes.DataDevice(),
			boxes.DataDevice());

		/*nodes.TransferToHost();

		for (auto id = 0; id < size - 1; ++id) {
			const auto node = nodes[id];
			std::cout << "Node" << " " << id << ": " << node.childLeft << " " << node.childRight << "        " << node.atomic << " " << node.rangeLeft << " " << node.rangeRight << std::endl;
		}*/
	}

}