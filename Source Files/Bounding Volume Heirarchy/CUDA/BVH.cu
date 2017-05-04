#include "BVH.cuh"
#include "Utility\CUDA\CUDAHelper.cuh"
#include "Utility/Logger.h"
#include "Algorithms\Morton Code\MortonCode.h"

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

// Returns the highest differing bit of i and i+1
__device__ uint HighestBit(MiniObject& objThis, uint64 mortonThis, MiniObject& objNext, uint64 mortonNext)
{

	int min = glm::min(objThis.tSize, objNext.tSize);
	int max = glm::min(objThis.tSize, objNext.tSize);

	uint bitDiff = 0;
	uint bitCount = (max-min)*64;

	for (int t = 0; t < min; ++t) {
		bitCount += 64;
		uint64 change = objThis.transforms[t].morton^objNext.transforms[t].morton;
		bitDiff += __clzll(change);
	}

	return bitCount-bitDiff;
}

//used for a given offset
__device__ void HighestBitOffset(MiniObject& objThis, uint64 mortonThis, MiniObject& objNext, uint64 mortonNext, uint& offset)
{

	int min = glm::min(objThis.tSize, objNext.tSize);
	int max = glm::min(objThis.tSize, objNext.tSize);

	uint bitDiff = 0;
	uint bitCount = (max - min) * 64;

	for (int t = 0; t < min; ++t) {
		bitCount += 64;
		uint64 change = objThis.transforms[t].morton^objNext.transforms[t].morton;
		bitDiff += __clzll(change);
	}

	//return bitCount - bitDiff;


}

__global__ void BuildTree(const uint n, BVHData* data, Node* nodes, Face* faces, Vertex* vertices, MiniObject* objects, const uint leafOffset)
{
	uint index = getGlobalIdx_1D_1D();
	if (index >= n)
		return;

	Node* currentNode = nodes + (leafOffset + index);

	while (true) {
		// Allow only one thread to process a node
		if (atomicAdd(&(currentNode->atomic), 1) != 1)
			return;

		// Set transform if the node is not a leaf
		if (currentNode - nodes < leafOffset)
		{
			currentNode->childLeft->parent = currentNode;
			currentNode->childRight->parent = currentNode;

			uint64 mortLeft = currentNode->childLeft->morton;
			uint64 mortRight = currentNode->childRight->morton;

			glm::vec3 posLeft = MortonCode::DecodeMorton(mortLeft);
			glm::vec3 posRight = MortonCode::DecodeMorton(mortRight);

			uint leadingZeros = _lzcnt_u64(mortLeft^mortRight);
			currentNode->morton = mortRight >> (64 - leadingZeros);

			currentNode->box.max = glm::max(currentNode->childLeft->box.max, currentNode->childRight->box.max);
			currentNode->box.min = glm::min(currentNode->childLeft->box.min, currentNode->childRight->box.min);

			currentNode->transform = glm::mat4();
		}

		uint left = currentNode->rangeLeft;
		uint right = currentNode->rangeRight;

		if (left == 0 && right == leafOffset) {
			data->root = currentNode;
			return;
		}


		Node* parent;
		Face a = faces[nodes[leafOffset + left - 1].faceID];
		Face b = faces[nodes[leafOffset + left].faceID];
		Face c = faces[nodes[leafOffset + right].faceID];
		Face d = faces[nodes[leafOffset + right + 1].faceID];
		if (left == 0 || (right < leafOffset && HighestBit(
			objects[vertices[a.indices.x].object], a.mortonCode,
			objects[vertices[b.indices.x].object], b.mortonCode) >
			HighestBit(
				objects[vertices[c.indices.x].object], c.mortonCode,
				objects[vertices[d.indices.x].object], d.mortonCode)))
		{
			// parent = right, set parent left child and range to node
			parent = nodes + right;
			parent->childLeft = currentNode;
			parent->rangeLeft = left;

		}
		else
		{
			// parent = left -1, set parent right child and range to node
			parent = nodes + (left - 1);
			parent->childRight = currentNode;
			parent->rangeRight = right;
		}



		currentNode = parent;
	}
}


__global__ void Reset(const uint n, BVHData* data, Node* nodes, Face* faces, Vertex* vertices, const uint leafOffset)
{
	uint index = getGlobalIdx_1D_1D();

	if (index >= n) {
		return;
	}

	Node temp;

	// Set ranges
	temp.rangeLeft = index;
	temp.rangeRight = index;
	temp.atomic = 1; // To allow the next thread to process
	temp.childLeft = nullptr;
	temp.childRight = nullptr;
	temp.parent = nullptr;
	temp.transform = glm::mat4();
	//store the camera
	if (index < leafOffset) {
		Node tempF;

		tempF.atomic = 0;
		tempF.childLeft = nodes + leafOffset + index;
		tempF.childRight = nodes + leafOffset + index + 1;

		nodes[index] = tempF;
	}

	// Set triangles in leaf
	temp.faceID = index;
	temp.morton = faces[index].mortonCode;
	glm::uvec3 ind = faces[index].indices;

	// Expand bounds using min/max functions
	glm::vec3 pos0 = vertices[ind.x].position;

	temp.bestObjID = vertices[ind.x].object;

	glm::vec3 max = pos0;
	glm::vec3 min = pos0;

	glm::vec3 pos1 = vertices[ind.y].position;
	max = glm::max(pos1, max);
	min = glm::min(pos1, min);

	glm::vec3 pos2 = vertices[ind.z].position;
	max = glm::max(pos2, max);
	min = glm::min(pos2, min);

	temp.box.max = max;
	temp.box.min = min;

	//set camera
	if (faces[index].isCamera) {
		data->camera = nodes + leafOffset + index;
	}
	nodes[leafOffset + index] = temp;

	// Special case
	if (n == 1)
	{
		nodes[0].box = nodes[leafOffset + 0].box;
		nodes[0].childLeft = nodes + leafOffset + 0;
	}
}

void BVH::Build(uint size, BVHData*& data, Face * faces, Vertex * vertices, MiniObject* objects) {

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

		Reset << <gridSize, blockSize >> > (size, data, bvh, faces, vertices, size - 1);
		CudaCheck(cudaPeekAtLastError());
		CudaCheck(cudaDeviceSynchronize());

		BuildTree << <gridSize, blockSize >> > (size, data, bvh, faces, vertices, objects, size - 1);
		CudaCheck(cudaPeekAtLastError());
		CudaCheck(cudaDeviceSynchronize());
	}

}