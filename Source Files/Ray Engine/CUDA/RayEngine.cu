#include "RayEngine.cuh"

#define STACK_SIZE 64
#define DYNAMIC_FETCH_THRESHOLD 20          // If fewer than this active, fetch new rays
#define RAY_BIAS_DISTANCE 0.0002f 

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <cuda_runtime.h>
#include "Utility\CUDA\CUDAHelper.cuh"
#include "GPGPU\CUDA\CUDABackend.h"
#include "Utility\Logger.h"

#include <curand_kernel.h>

Ray* deviceRays = nullptr;
Ray* deviceRaysB = nullptr;
RayJob* jobs = nullptr;
Scene* scene = nullptr;
curandState* randomState = nullptr;

uint raySeedGl = 0;
uint raysAllocated = 0;
uint jobsAllocated = 0;

const uint rayDepth = 4;

//stored counters
int* counter;
int* hitAtomic;

//engine launch config
uint blockCountE;
dim3 blockSizeE;
uint warpPerBlock;

template <class T> __host__ __device__ __inline__ void swap(T& a, T& b)
{
	T t = a;
	a = b;
	b = t;
}

__device__ __inline__ int __float_as_int(float in) {
	union fi { int i; float f; } conv;
	conv.f = in;
	return conv.i;
}

__device__ __inline__ float __int_as_float(int a)

{

	union { int a; float b; } u;

	u.a = a;

	return u.b;

}

__device__ __inline__ float min4(float a, float b, float c, float d)
{
	return fminf(fminf(fminf(a, b), c), d);
}

__device__ __inline__ float max4(float a, float b, float c, float d)
{
	return fmaxf(fmaxf(fmaxf(a, b), c), d);
}

__device__ __inline__ float min3(float a, float b, float c)
{
	return fminf(fminf(a, b), c);
}

__device__ __inline__ float max3(float a, float b, float c)
{
	return fmaxf(fmaxf(a, b), c);
}

// Using integer min,max
__inline__ __device__ float fminf2(float a, float b)
{
	int a2 = __float_as_int(a);
	int b2 = __float_as_int(b);
	return __int_as_float(a2 < b2 ? a2 : b2);
}

__inline__ __device__ float fmaxf2(float a, float b)
{
	int a2 = __float_as_int(a);
	int b2 = __float_as_int(b);
	return __int_as_float(a2 > b2 ? a2 : b2);
}

// Using video instructions
__device__ __inline__ int   min_min(int a, int b, int c) {
	int v;
	asm("vmin.s32.s32.s32.min %0, %1, %2, %3;"
		: "=r"(v) : "r"(a), "r"(b), "r"(c));
	return v;
}
__device__ __inline__ int   min_max(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_min(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_max(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ float fmin_fmin(float a, float b, float c) { return __int_as_float(min_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmin_fmax(float a, float b, float c) { return __int_as_float(min_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmin(float a, float b, float c) { return __int_as_float(max_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmax(float a, float b, float c) { return __int_as_float(max_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }


__device__ __inline__ float magic_max7(float a0, float a1, float b0, float b1, float c0, float c1, float d)
{
	float t1 = fmin_fmax(a0, a1, d);
	float t2 = fmin_fmax(b0, b1, t1);
	float t3 = fmin_fmax(c0, c1, t2);
	return t3;
}

__device__ __inline__ float magic_min7(float a0, float a1, float b0, float b1, float c0, float c1, float d)
{
	float t1 = fmax_fmin(a0, a1, d);
	float t2 = fmax_fmin(b0, b1, t1);
	float t3 = fmax_fmin(c0, c1, t2);
	return t3;
}

__device__ __inline__ float spanBeginKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d) { return fmax_fmax(fminf(a0, a1), fminf(b0, b1), fmin_fmax(c0, c1, d)); }
__device__ __inline__ float spanEndKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d) { return fmin_fmin(fmaxf(a0, a1), fmaxf(b0, b1), fmax_fmin(c0, c1, d)); }


__host__ __device__ __inline__ uint WangHash(uint a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

__global__ void RandomSetup(const uint n, curandState* randomState, const uint raySeed) {


	uint index = getGlobalIdx_1D_1D();

	if (index >= n) {
		return;
	}

	curandState randState;
	curand_init(index, 0, 0, &randState);

	randomState[index] = randState;

}

__global__ void EngineSetup(const uint n, RayJob* jobs, int jobSize) {


	uint index = getGlobalIdx_1D_1D();

	if (index >= n) {
		return;
	}

	uint startIndex = 0;

	int cur = 0;

	((glm::vec4*)jobs[cur].results)[(index - startIndex)] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);

}

__global__ void RaySetup(const uint n, RayJob* job, int jobSize, Ray* rays, const Scene* scene, curandState* randomState) {

	uint index = getGlobalIdx_1D_1D();

	if (index >= n) {
		return;
	}

	uint startIndex = 0;
	int cur = 0;

	uint localIndex = (index - startIndex) / job[cur].samples;

	curandState randState = randomState[index];

	Ray ray;
	ray.storage = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
	ray.resultOffset = localIndex;
	job[cur].camera.SetupRay(localIndex, ray, randState);

	randomState[index] = randState;
	rays[index] = ray;
}

__host__ __device__ __inline__ glm::vec3 PositionAlongRay(const Ray& ray, const float& t) {
	return glm::vec3(ray.origin.x, ray.origin.y, ray.origin.z) + t * glm::vec3(ray.direction.x, ray.direction.y, ray.direction.z);
}

__host__ __device__ __inline__ bool FindTriangleIntersect(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c,
	const glm::vec3& rayO, const glm::vec3& rayD, const glm::vec3& invDir,
	float& t, const float& tMax, float& bary1, float& bary2)
{

	glm::vec3 edge1 = b - a;
	glm::vec3 edge2 = c - a;

	glm::vec3 pvec = glm::cross(rayD, edge2);

	float det = glm::dot(edge1, pvec);

	if (det == 0.f) {
		return false;
	}

	float inv_det = 1.0f / det;

	glm::vec3 tvec = rayO - a;

	bary1 = glm::dot(tvec, pvec) * inv_det;

	glm::vec3 qvec = glm::cross(tvec, edge1);

	bary2 = glm::dot(rayD, qvec) * inv_det;

	t = glm::dot(edge2, qvec) * inv_det;

	return(t > EPSILON &&t < tMax && (bary1 >= 0.0f && bary2 >= 0.0f && (bary1 + bary2) <= 1.0f));
}


__host__ __device__ bool AABBIntersect(const BoundingBox& box, const glm::vec3& o, const glm::vec3& dInv, const float& t0, float& t1) {

	float temp1 = (box.min.x - o.x)*dInv.x;
	float temp2 = (box.max.x - o.x)*dInv.x;

	float tMin = glm::min(temp1, temp2);
	float tmax = glm::max(temp1, temp2);

	temp1 = (box.min.y - o.y)*dInv.y;
	temp2 = (box.max.y - o.y)*dInv.y;

	tMin = glm::max(tMin, glm::min(temp1, temp2));
	tmax = glm::min(tmax, glm::max(temp1, temp2));

	temp1 = (box.min.z - o.z)*dInv.z;
	temp2 = (box.max.z - o.z)*dInv.z;

	tMin = glm::max(tMin, glm::min(temp1, temp2));
	tmax = glm::min(tmax, glm::max(temp1, temp2));

	float tTest = t1;
	t1 = tMin;
	return tmax >= glm::max(t0, tMin) && tMin < tTest;
}


__global__ void ProcessHits(const uint n, RayJob* job, int jobSize, Ray* rays, Ray* raysNew, const Scene* scene, int * nAtomic, curandState* randomState) {

	uint index = getGlobalIdx_1D_1D();

	if (index >= n) {
		return;
	}

	int cur = 0;


	Ray ray = rays[index];

	uint faceHit = ray.currentHit;
	float hitT = ray.direction.w;

	uint localIndex = ray.resultOffset;

	curandState randState = randomState[index];

	glm::vec3 col;

	if (faceHit == uint(-1)) {

		col = glm::vec3(ray.storage.x, ray.storage.y, ray.storage.z)*scene->sky->ExtractColour({ ray.direction.x, ray.direction.y, ray.direction.z });

	}
	else {

		Face face = scene->faces[ray.currentHit];

		Material mat = scene->materials[face.material];

		Vertex* vP = scene->vertices;

		glm::vec2 bary = ray.bary;

		glm::vec3 n0 = vP[face.indices.x].position;
		glm::vec3 n1 = vP[face.indices.y].position;
		glm::vec3 n2 = vP[face.indices.z].position;

		glm::vec3 bestNormal = glm::normalize(glm::cross(n1 - n0, n2 - n0));

		/*glm::vec3 n0 = vP[faceHit->indices.x].normal;
		glm::vec3 n1 = vP[faceHit->indices.y].normal;
		glm::vec3 n2 = vP[faceHit->indices.z].normal;

		glm::vec3 bestNormal = (1 - bary.x - bary.y) * n0 + bary.x * n1 + bary.y * n2;*/

		glm::vec2 uv0 = vP[face.indices.x].textureCoord;
		glm::vec2 uv1 = vP[face.indices.y].textureCoord;
		glm::vec2 uv2 = vP[face.indices.z].textureCoord;

		glm::vec2 bestUV = (1.0f - bary.x - bary.y) * uv0 + bary.x * uv1 + bary.y * uv2;


		glm::vec3 biasVector = (RAY_BIAS_DISTANCE * bestNormal);

		glm::vec3 bestIntersectionPoint = PositionAlongRay(ray, hitT);


		glm::vec3 accumulation;
		accumulation.x = ray.storage.x * mat.emit.x;
		accumulation.y = ray.storage.y * mat.emit.y;
		accumulation.z = ray.storage.z * mat.emit.z;

		//tex2D<float>(texObj, tu, tv)
		//unsigned char blue = tex2D<unsigned char>(mat->texObj, (4 * localIndex), localIndex);
		//unsigned char green = tex2D<unsigned char>(mat->texObj, (4 * localIndex) + 1, localIndex);
		//unsigned char red = tex2D<unsigned char>(mat->texObj, (4 * localIndex) + 2, localIndex);

		float4 PicCol = tex2DLod<float4>(mat.diffuseImage.texObj, bestUV.x * 20, bestUV.y * 20, 0.0f);
		//float PicCol = tex2D<float>(mat->texObj, bestUV.x * 50, bestUV.y * 50);
		ray.storage *= glm::vec4(PicCol.x, PicCol.y, PicCol.z, 1.0f);


		//ray.storage *= mat->diffuse;

		glm::vec3 orientedNormal = glm::dot(bestNormal, glm::vec3(ray.direction.x, ray.direction.y, ray.direction.z)) < 0 ? bestNormal : bestNormal * -1.0f;

		float r1 = 2 * PI * curand_uniform(&randState);
		float r2 = curand_uniform(&randState);
		float r2s = sqrtf(r2);

		glm::vec3 u = glm::normalize(glm::cross((glm::abs(orientedNormal.x) > .1 ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0)), orientedNormal));
		glm::vec3 v = glm::cross(orientedNormal, u);

		ray.origin = glm::vec4(bestIntersectionPoint + biasVector, 0.0f);
		ray.direction = glm::vec4(glm::normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + orientedNormal*sqrtf(1 - r2)), 40000000000000.0f);

		col = accumulation;

		raysNew[FastAtomicAdd(nAtomic)] = ray;

		//update the state
		randomState[index] = randState;
	}
	col /= job[cur].samples;

	glm::vec4* pt = &((glm::vec4*)job[cur].results)[localIndex];

	atomicAdd(&(pt->x), col.x);

	atomicAdd(&(pt->y), col.y);

	atomicAdd(&(pt->z), col.z);


}




__global__ void EngineExecute(const uint n, RayJob* job, int jobSize, Ray* rays, const Scene* scene, int* counter) {


	Node * traversalStack[STACK_SIZE];
	traversalStack[0] = nullptr; // Bottom-most entry.


	float   origx, origy, origz;            // Ray origin.
	char    stackPtr;                       // Current position in traversal stack.
	Node*   currentLeaf;                       // First postponed leaf, non-negative if none.
	Node*   currentNode = nullptr;				// Non-negative: current internal node, negative: second postponed leaf.
	int     rayidx;
	float   oodx;
	float   oody;
	float   oodz;
	float   dirx;
	float   diry;
	float   dirz;
	float   idirx;
	float   idiry;
	float   idirz;

	//scene pointers
	Ray ray;
	BVHData bvh = *(scene->bvhData);
	Vertex* vP = scene->vertices;
	Face* fP = scene->faces;

	extern __shared__ volatile int nextRayArray[]; // Current ray index in global buffer needs the (max) block height. 

	do {
		const uint tidx = threadIdx.x;
		volatile int& rayBase = nextRayArray[threadIdx.y];


		const bool          terminated = currentNode == nullptr;
		const unsigned int  maskTerminated = __ballot(terminated);
		const int           numTerminated = __popc(maskTerminated);
		const int           idxTerminated = __popc(maskTerminated & ((1u << tidx) - 1));

		//fetch a new ray

		if (terminated)
		{
			if (idxTerminated == 0) {
				rayBase = atomicAdd(counter, numTerminated);
			}

			rayidx = rayBase + idxTerminated;
			if (rayidx >= n) {
				break;
			}

			ray = rays[rayidx];
			origx = ray.origin.x;
			origy = ray.origin.y;
			origz = ray.origin.z;
			dirx = ray.direction.x;
			diry = ray.direction.y;
			dirz = ray.direction.z;
			float ooeps = exp2f(-80.0f); // Avoid div by zero.
			idirx = 1.0f / (fabsf(dirx) > ooeps ? dirx : copysignf(ooeps, dirx));
			idiry = 1.0f / (fabsf(diry) > ooeps ? diry : copysignf(ooeps, diry));
			idirz = 1.0f / (fabsf(dirz) > ooeps ? dirz : copysignf(ooeps, dirz));
			oodx = origx * idirx;
			oody = origy * idiry;
			oodz = origz * idirz;

			// Setup traversal.

			stackPtr = 0;
			currentLeaf = nullptr;   // No postponed leaf.
			currentNode = bvh.root;   // Start from the root.
			ray.currentHit = -1;  // No triangle intersected so far.
		}

		//Traversal starts here

		while (currentNode != nullptr)
		{
			// Until all threads find a leaf, traverse

			while (!bvh.IsLeaf(currentNode) && currentNode != nullptr)
			{
				// Fetch AABBs of the two child nodes.

				Node* childL = currentNode->childLeft;
				Node* childR = currentNode->childRight;

				glm::vec3  b0Min = childL->box.min;
				glm::vec3  b0Max = childL->box.max;

				glm::vec3  b1Min = childR->box.min;
				glm::vec3  b1Max = childR->box.max;

				// Intersect the ray against the child nodes.

				const float c0lox = b0Min.x * idirx - oodx;
				const float c0hix = b0Max.x * idirx - oodx;
				const float c0loy = b0Min.y * idiry - oody;
				const float c0hiy = b0Max.y * idiry - oody;
				const float c0loz = b0Min.z   * idirz - oodz;
				const float c0hiz = b0Max.z   * idirz - oodz;
				const float c1loz = b1Min.z   * idirz - oodz;
				const float c1hiz = b1Max.z   * idirz - oodz;
				const float c0min = spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, ray.origin.w);
				const float c0max = spanEndKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, ray.direction.w);
				const float c1lox = b1Min.x * idirx - oodx;
				const float c1hix = b1Max.x * idirx - oodx;
				const float c1loy = b1Min.y * idiry - oody;
				const float c1hiy = b1Max.y * idiry - oody;
				const float c1min = spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, ray.origin.w);
				const float c1max = spanEndKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, ray.direction.w);

				bool swp = (c1min < c0min);

				bool traverseChild0 = (c0max >= c0min);
				bool traverseChild1 = (c1max >= c1min);


				if (!traverseChild0 && !traverseChild1)
				{
					currentNode = traversalStack[stackPtr--];
				}

				// Otherwise => fetch child pointers.

				else
				{
					currentNode = (traverseChild0) ? childL : childR;

					// Both children were intersected => push the farther one.

					if (traverseChild0 && traverseChild1)
					{
						if (swp) {
							swap(currentNode, childR);
						}

						traversalStack[++stackPtr] = childR;
					}
				}

				// First leaf => postpone and continue traversal.

				if (bvh.IsLeaf(currentNode) && !bvh.IsLeaf(currentLeaf))     // Postpone leaf
				{
					currentLeaf = currentNode;
					currentNode = traversalStack[stackPtr--];
				}


				if (!__any(!bvh.IsLeaf(currentLeaf))) {
					break;
				}
			}

			// Process postponed leaf nodes.

			while (bvh.IsLeaf(currentLeaf))
			{
				uint faceID = currentLeaf->faceID;

				glm::uvec3 face = fP[faceID].indices;

				float bary1;
				float bary2;
				float tTemp;

				if (FindTriangleIntersect(vP[face.x].position, vP[face.y].position, vP[face.z].position,
				{ origx, origy, origz, }, { dirx, diry, dirz }, { idirx, idiry, idirz },
					tTemp, ray.direction.w, bary1, bary2)) {

					ray.direction.w = tTemp;
					ray.bary = glm::vec2(bary1, bary2);
					ray.currentHit = faceID;
				}

				//go through the second postponed leaf
				currentLeaf = currentNode;
				if (bvh.IsLeaf(currentNode))
				{
					currentNode = traversalStack[stackPtr--];
				}
			}

			//cut the losses

			if (__popc(__ballot(true)) < DYNAMIC_FETCH_THRESHOLD) {
				break;
			}

		} // traversal

		//update the data

		rays[rayidx] = ray;

	} while (true);
}

//grab the shared memory for a dynamic blocksize
__host__ int ReturnSharedBytes(int blockSize) {
	return blockSize / CUDABackend::GetWarpSize() * sizeof(int);
}


__host__ void ProcessJobs(std::vector<RayJob>& hjobs, const Scene* sceneIn) {

	uint numberJobs = hjobs.size();

	//only upload data if a job exists
	if (numberJobs > 0) {

		uint numberResults = 0;
		uint numberRays = 0;
		uint samplesMax = 0;

		for (int i = 0; i < numberJobs; ++i) {

			hjobs[i].startIndex = numberResults;
			numberResults += hjobs[i].rayAmount;
			numberRays += hjobs[i].rayAmount* hjobs[i].samples;

			if (hjobs[i].samples > samplesMax) {
				samplesMax = hjobs[i].samples;
			}

		}

		if (numberResults != 0) {

			if (numberJobs > jobsAllocated) {

				if (jobs) {
					CudaCheck(cudaFree(jobs));
				}

				CudaCheck(cudaMalloc((void**)&jobs, numberJobs * sizeof(RayJob)));

				jobsAllocated = numberJobs;

			}

			//copy device jobs
			CudaCheck(cudaMemcpy(jobs, hjobs.data(), numberJobs * sizeof(RayJob), cudaMemcpyHostToDevice));

			//remove all the jobs as they are transfered
			hjobs.clear();

			uint blockSize = 64;
			uint blockCount = (numberResults + blockSize - 1) / blockSize;


			//clear the jobs result memory, required for accumulation of multiple samples
			EngineSetup << <blockCount, blockSize >> > (numberResults, jobs, numberJobs);
			CudaCheck(cudaPeekAtLastError());
			CudaCheck(cudaDeviceSynchronize());

			blockSize = 64;
			blockCount = (numberRays + blockSize - 1) / blockSize;

			if (numberRays != 0) {


				if (numberRays > raysAllocated) {

					if (deviceRays) {
						CudaCheck(cudaFree(deviceRays));
					}
					if (deviceRaysB) {
						CudaCheck(cudaFree(deviceRaysB));
					}
					if (randomState) {
						CudaCheck(cudaFree(randomState));
					}

					CudaCheck(cudaMalloc((void**)&randomState, numberRays * sizeof(curandState)));
					CudaCheck(cudaMalloc((void**)&deviceRays, numberRays * sizeof(Ray)));
					CudaCheck(cudaMalloc((void**)&deviceRaysB, numberRays * sizeof(Ray)));

					RandomSetup << <blockCount, blockSize >> > (numberRays, randomState, WangHash(++raySeedGl));
					CudaCheck(cudaPeekAtLastError());
					CudaCheck(cudaDeviceSynchronize());

					raysAllocated = numberRays;

				}

				//copy the scene over
				CudaCheck(cudaMemcpy(scene, sceneIn, sizeof(Scene), cudaMemcpyHostToDevice));

				RaySetup << <blockCount, blockSize >> > (numberRays, jobs, numberJobs, deviceRays, scene, randomState);
				CudaCheck(cudaPeekAtLastError());
				CudaCheck(cudaDeviceSynchronize());


				//setup the counters
				int zeroHost = 0;

				//start the engine loop
				uint numActive = numberRays;

				for (uint i = 0; i < rayDepth && numActive>0; ++i) {

					//reset counters
					CudaCheck(cudaMemcpy(hitAtomic, &zeroHost, sizeof(int), cudaMemcpyHostToDevice));
					CudaCheck(cudaMemcpy(counter, &zeroHost, sizeof(int), cudaMemcpyHostToDevice));

					//grab the current block sizes for collecting hits based on numActive
					blockSize = 64;
					blockCount = (numActive + blockSize - 1) / blockSize;

					//main engine, collects hits
					EngineExecute << <blockCountE, blockSizeE, warpPerBlock*sizeof(int) >> > (numActive, jobs, numberJobs, deviceRays, scene, counter);
					CudaCheck(cudaPeekAtLastError());
					CudaCheck(cudaDeviceSynchronize());

					//processes hits 
					ProcessHits << <blockCount, blockSize >> > (numActive, jobs, numberJobs, deviceRays, deviceRaysB, scene, hitAtomic, randomState);
					CudaCheck(cudaPeekAtLastError());
					CudaCheck(cudaDeviceSynchronize());

					swap(deviceRays, deviceRaysB);

					CudaCheck(cudaMemcpy(&numActive, hitAtomic, sizeof(int), cudaMemcpyDeviceToHost));

				}
			}
		}
	}
}

__host__ void GPUInitialize() {

	CudaCheck(cudaMalloc((void**)&scene, sizeof(Scene)));
	CudaCheck(cudaMalloc((void**)&counter, sizeof(int)));
	CudaCheck(cudaMalloc((void**)&hitAtomic, sizeof(int)));

	int blockSizeOut;
	int minGridSize;

	//dynamically ask for the best launch setup
	CudaCheck(cudaOccupancyMaxPotentialBlockSizeVariableSMem(
		&minGridSize, &blockSizeOut, EngineExecute, ReturnSharedBytes,
		CUDABackend::GetSMCount()*CUDABackend::GetBlocksPerMP()));

	//get the device stats for persistant threads
	uint warpSize = CUDABackend::GetWarpSize();
	warpPerBlock = blockSizeOut / warpSize;
	blockCountE = minGridSize;
	blockSizeE = dim3(warpSize, warpPerBlock, 1);

	///////////////Alternative Hardcoded Calculation/////////////////
	//uint blockPerSM = CUDABackend::GetBlocksPerMP();
	//warpPerBlock = CUDABackend::GetWarpsPerMP() / blockPerSM;
	//blockCountE = CUDABackend::GetSMCount()*blockPerSM;
	//blockSizeE = dim3(CUDABackend::GetWarpSize(), warpPerBlock, 1);

}

__host__ void GPUTerminate() {

	CudaCheck(cudaFree(scene));
	CudaCheck(cudaFree(counter));
	CudaCheck(cudaFree(hitAtomic));
	CudaCheck(cudaFree(jobs));
	CudaCheck(cudaFree(deviceRays));
	CudaCheck(cudaFree(deviceRaysB));
	CudaCheck(cudaFree(randomState));

}