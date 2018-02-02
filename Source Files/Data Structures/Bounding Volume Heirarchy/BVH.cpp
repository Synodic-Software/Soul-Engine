#include "BVH.h"
#include "CUDA/BVH.cuh"
#include "Compute/ComputeManager.h"
#include "Compute\DeviceAPI.h"

BVH::BVH() :
	bvh(S_BEST_GPU)
{

}

__host__ __device__ inline void Reset(uint index, uint n, uint innerSize, Node* nodes, Face* faces, Vertex* vertices)
{

	const uint leafOffset = innerSize + index;

	//set the inner node
	if (index < innerSize) {
		Node temp;
		temp.atomic = 0; //inner nodes are not visited
		temp.childLeft = leafOffset;
		temp.childRight = leafOffset + 1;
		nodes[index] = temp;
	}


	const glm::uvec3 ind = faces[index].indices;

	// Expand bounds using min/max functions
	const glm::vec3 pos0 = vertices[ind.x].position;
	const glm::vec3 pos1 = vertices[ind.y].position;
	const glm::vec3 pos2 = vertices[ind.z].position;

	glm::vec3 max = pos0;
	glm::vec3 min = pos0;

	max = glm::max(pos1, max);
	min = glm::min(pos1, min);

	max = glm::max(pos2, max);
	min = glm::min(pos2, min);

	//set the leaf node
	Node temp;
	temp.rangeLeft = index;
	temp.rangeRight = index;
	temp.atomic = 1; // To allow the next thread to process
	temp.faceID = index; //set triangle
	temp.box.max = max;
	temp.box.min = min;
	temp.childLeft = static_cast<uint>(-1); //set termination
	temp.childRight = static_cast<uint>(-1); //set termination

	nodes[leafOffset] = temp;
}

void BVH::Build(int size, ComputeBuffer<BVHData>& data, ComputeBuffer<uint64>& mortonCodes, ComputeBuffer<Face>& faces, ComputeBuffer<Vertex>& vertices) {

	if (size > 0) {

		const int nodeSize = size * 2 - 1;

		bvh.ResizeDevice(nodeSize);

		data[0].currentSize = size;
		data[0].bvh = bvh.DataDevice();
		data.TransferToDevice();

		ComputeDevice device = S_BEST_GPU;
		ComputeDevice device2 = S_BEST_CPU;

		const GPUExecutePolicy normalPolicy(size, 64, 0, 0);

		device.Launch(normalPolicy, size, Reset,
			size - 1,
			bvh.DataDevice(),
			faces.DataDevice(),
			vertices.DataDevice());

		device.LaunchOld(normalPolicy, BuildTree,
			size,
			size - 1,
			data.DataDevice(),
			bvh.DataDevice(),
			mortonCodes.DataDevice());

	}

}