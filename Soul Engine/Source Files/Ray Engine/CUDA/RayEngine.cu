#include "RayEngine.cuh"

#define STACK_SIZE 64
#define DYNAMIC_FETCH_THRESHOLD 20          // If fewer than this active, fetch new rays
#define RAY_BIAS_DISTANCE 0.0002f 

struct is_marked
{
	__host__ __device__
		bool operator()(const Ray& x)
	{
		return (x.currentHit == NULL);
	}
};

Ray* deviceRays = NULL;
Ray* deviceRaysB = NULL;
uint raySeedGl = 0;
uint numRaysAllocated = 0;
const uint rayDepth = 3;


template <class T> CUDA_FUNCTION __inline__ void swap(T& a, T& b)
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

__device__ __inline__ float spanBeginKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d){ return fmax_fmax(fminf(a0, a1), fminf(b0, b1), fmin_fmax(c0, c1, d)); }
__device__ __inline__ float spanEndKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d)	{ return fmin_fmin(fmaxf(a0, a1), fmaxf(b0, b1), fmax_fmin(c0, c1, d)); }


CUDA_FUNCTION __inline__ uint WangHash(uint a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

__device__ __inline__ int GetCurrentJob(RayJob** jobs, int& jobSize, const uint& index, const uint& startIndex){

	int i = 0;
	for (;
		i < jobSize && !(index < startIndex + jobs[i]->GetRayAmount()*jobs[i]->GetSampleAmount());
		++i){

	}
	return i;
	//while (job->nextRay != NULL && !(index < startIndex + job->rayAmount*job->samples)){
	//	startIndex += job->rayAmount*job->samples;
	//	job = job->nextRay;
	//}

}
__device__ __inline__ int GetCurrentJobNoSample(RayJob** jobs, int& jobSize, const uint& index, const uint& startIndex){

	int i = 0;
	for (;
		i < jobSize && !(index < startIndex + jobs[i]->GetRayAmount());
		++i){

	}
	return i;
	//while (job->nextRay != NULL && !(index < startIndex + job->rayAmount*job->samples)){
	//	startIndex += job->rayAmount*job->samples;
	//	job = job->nextRay;
	//}

}

__global__ void EngineResultClear(const uint n, RayJob** jobs, int jobSize){


	uint index = getGlobalIdx_1D_1D();

	if (index >= n){
		return;
	}

	uint startIndex = 0;

	int cur = GetCurrentJobNoSample(jobs, jobSize, index, startIndex);

	((glm::vec4*)jobs[cur]->GetResultPointer(0))[(index - startIndex)] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
}

__global__ void RaySetup(const uint n, RayJob** job, int jobSize, Ray* rays, const uint raySeed, const Scene* scene){

	uint index = getGlobalIdx_1D_1D();

	if (index >= n){
		return;
	}

	uint startIndex = 0;
	int cur = GetCurrentJob(job, jobSize, index, startIndex);


	//thrust::default_random_engine rng(randHash(raySeed) * randHash(index));
	//thrust::uniform_real_distribution<float> uniformDistribution(0.0f, 1.0f);


	curandState randState;
	curand_init(raySeed + index, 0, 0, &randState);

	uint localIndex = (index - startIndex) / job[cur]->GetSampleAmount();

	Ray ray;
	ray.storage = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
	ray.resultOffset = localIndex;
	job[cur]->GetCamera()->SetupRay(localIndex, ray, randState);

	rays[index] = ray;
}

CUDA_FUNCTION __inline__ glm::vec3 PositionAlongRay(const Ray& ray, const float& t) {
	return glm::vec3(ray.origin.x, ray.origin.y, ray.origin.z) + t * glm::vec3(ray.direction.x, ray.direction.y, ray.direction.z);
}
CUDA_FUNCTION __inline__ glm::vec3 computeBackgroundColor(const glm::vec3& direction) {
	float position = (glm::dot(direction, normalize(glm::vec3(-0.5, 0.5, -1.0))) + 1) / 2.0f;
	return (1.0f - position) * glm::vec3(0.5f, 0.5f, 1.0f) + position * glm::vec3(0.7f, 0.7f, 1.0f);
	//return glm::vec3(0.0f, 0.0f, 0.0f);
}

CUDA_FUNCTION __inline__ bool FindTriangleIntersect(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c,
	const glm::vec3& rayO, const glm::vec3& rayD, const glm::vec3& invDir,
	float& t, const float& tMax, float& bary1, float& bary2)
{

	glm::vec3 edge1 = b - a;
	glm::vec3 edge2 = c - a;

	glm::vec3 pvec = glm::cross(rayD, edge2);

	float det = glm::dot(edge1, pvec);

	if (det == 0.f){
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


CUDA_FUNCTION bool AABBIntersect(const BoundingBox& box, const glm::vec3& o, const glm::vec3& dInv, const float& t0, float& t1){

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


__global__ void CollectHits(const uint n, RayJob** job, int jobSize, Ray* rays, Ray* raysNew, const uint raySeed, const Scene* scene, int * nAtomic){

	uint index = getGlobalIdx_1D_1D();

	if (index >= n){
		return;
	}

	uint startIndex = 0;
	int cur = GetCurrentJob(job, jobSize, index, startIndex);

	curandState randState;
	curand_init(raySeed + index, 0, 0, &randState);

	Ray ray = rays[index];

	Face* faceHit = ray.currentHit;
	float hitT = ray.direction.w;

	uint localIndex = ray.resultOffset;
	glm::vec3 col;

	if (faceHit == NULL){

		col = glm::vec3(ray.storage.x, ray.storage.y, ray.storage.z)*computeBackgroundColor(glm::vec3(ray.direction.x, ray.direction.y, ray.direction.z));

	}
	else{



		Material* mat = ray.currentHit->materialPointer;

		Vertex* vP = faceHit->objectPointer->vertices;

		glm::vec2 bary = ray.bary;

		glm::vec3 n0 = vP[faceHit->indices.x].position;
		glm::vec3 n1 = vP[faceHit->indices.y].position;
		glm::vec3 n2 = vP[faceHit->indices.z].position;

		glm::vec3 bestNormal = glm::normalize(glm::cross(n1 - n0, n2 - n0));

		/*glm::vec3 n0 = vP[faceHit->indices.x].normal;
		glm::vec3 n1 = vP[faceHit->indices.y].normal;
		glm::vec3 n2 = vP[faceHit->indices.z].normal;

		glm::vec3 bestNormal = (1 - bary.x - bary.y) * n0 + bary.x * n1 + bary.y * n2;*/

		glm::vec2 uv0 = vP[faceHit->indices.x].textureCoord;
		glm::vec2 uv1 = vP[faceHit->indices.y].textureCoord;
		glm::vec2 uv2 = vP[faceHit->indices.z].textureCoord;

		glm::vec2 bestUV = (1.0f - bary.x - bary.y) * uv0 + bary.x * uv1 + bary.y * uv2;


		glm::vec3 biasVector = (RAY_BIAS_DISTANCE * bestNormal);

		glm::vec3 bestIntersectionPoint = PositionAlongRay(ray, hitT);


		glm::vec3 accumulation;
		accumulation.x = ray.storage.x * mat->emit.x;
		accumulation.y = ray.storage.y * mat->emit.y;
		accumulation.z = ray.storage.z * mat->emit.z;

		//tex2D<float>(texObj, tu, tv)
		//unsigned char blue = tex2D<unsigned char>(mat->texObj, (4 * localIndex), localIndex);
		//unsigned char green = tex2D<unsigned char>(mat->texObj, (4 * localIndex) + 1, localIndex);
		//unsigned char red = tex2D<unsigned char>(mat->texObj, (4 * localIndex) + 2, localIndex);

		float4 PicCol = tex2DLod<float4>(mat->texObj, bestUV.x, bestUV.y, 0.0f);
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

		//newListwithatomics


		raysNew[FastAtomicAdd(nAtomic)] = ray;

	}
	col /= job[cur]->GetSampleAmount();

	//rays[index] = ray;

	glm::vec4* pt = &((glm::vec4*)job[cur]->GetResultPointer(0))[localIndex];

	atomicAdd(&(pt->x), col.x);

	atomicAdd(&(pt->y), col.y);

	atomicAdd(&(pt->z), col.z);


}




__global__ void EngineExecute(const uint n, RayJob** job, int jobSize, Ray* rays, const uint raySeed, const Scene* scene, int* counter){


	Node * traversalStack[STACK_SIZE];
	traversalStack[0] = NULL; // Bottom-most entry.


	float   origx, origy, origz;            // Ray origin.
	char    stackPtr;                       // Current position in traversal stack.
	Node*   currentLeaf;                       // First postponed leaf, non-negative if none.
	Node*   currentNode = NULL;				// Non-negative: current internal node, negative: second postponed leaf.
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

	Ray ray;

	__shared__ volatile int nextRayArray[BLOCK_HEIGHT]; // Current ray index in global buffer needs the (max) block height.

	do{
		const uint tidx = threadIdx.x;
		volatile int& rayBase = nextRayArray[threadIdx.y];


		const bool          terminated = currentNode == NULL;
		const unsigned int  maskTerminated = __ballot(terminated);
		const int           numTerminated = __popc(maskTerminated);
		const int           idxTerminated = __popc(maskTerminated & ((1u << tidx) - 1));

		//fetch a new ray

		if (terminated)
		{
			if (idxTerminated == 0){
				rayBase = atomicAdd(&(*counter), numTerminated);
			}

			rayidx = rayBase + idxTerminated;
			if (rayidx >= n){
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
			currentLeaf = NULL;   // No postponed leaf.
			currentNode = scene->bvh->GetRoot();   // Start from the root.
			ray.currentHit = NULL;  // No triangle intersected so far.
		}

		//Traversal starts here

		while (currentNode != NULL)
		{
			// Until all threads find a leaf, traverse

			while (!scene->bvh->IsLeaf(currentNode) && currentNode != NULL)
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
						if (swp){
							swap(currentNode, childR);
						}

						traversalStack[++stackPtr] = childR;
					}
				}

				// First leaf => postpone and continue traversal.

				if (scene->bvh->IsLeaf(currentNode) && !scene->bvh->IsLeaf(currentLeaf))     // Postpone leaf
				{
					currentLeaf = currentNode;
					currentNode = traversalStack[stackPtr--];
				}

				// Once all found, break to the processing

				if (!__any(!scene->bvh->IsLeaf(currentLeaf))){
					break;
				}
			}


			// Process postponed leaf nodes.

			while (scene->bvh->IsLeaf(currentLeaf))
			{
				// Tris in TEX (good to fetch as a single batch)
				glm::uvec3 face = currentLeaf->faceID->indices;

				float bary1;
				float bary2;
				float tTemp;

				Vertex* vP = currentLeaf->faceID->objectPointer->vertices;

				if (FindTriangleIntersect(vP[face.x].position, vP[face.y].position, vP[face.z].position,
				{ origx, origy, origz, }, { dirx, diry, dirz }, { idirx, idiry, idirz },
				tTemp, ray.direction.w, bary1, bary2)){

					ray.direction.w = tTemp;
					ray.bary = glm::vec2(bary1, bary2);
					ray.currentHit = currentLeaf->faceID;
				}

				//go throught the second postponed leaf
				currentLeaf = currentNode;
				if (scene->bvh->IsLeaf(currentNode))
				{
					currentNode = traversalStack[stackPtr--];
				}
			}

			//cut the losses

			if (__popc(__ballot(true)) < DYNAMIC_FETCH_THRESHOLD)
				break;

		} // traversal

		//update the data

		rays[rayidx] = ray;
		//__syncthreads();

	} while (true);
}


__host__ void ClearResults(std::vector<RayJob*>& hjobs){

	CudaCheck(cudaDeviceSynchronize());
	RayJob** jobs;
	uint jobsSize = hjobs.size();
	cudaMallocManaged((void**)&jobs, jobsSize*sizeof(RayJob*));

	CudaCheck(cudaDeviceSynchronize());

	for (int i = 0; i < jobsSize; i++){
	/*	std::cout << jobs[i] << std::endl;
		std::cout << hjobs[i] << std::endl;*/

		jobs[i] = hjobs[i];
	}

	CudaCheck(cudaDeviceSynchronize());

	if (jobsSize > 0){

		uint n = 0;
		for (int i = 0; i < jobsSize; ++i){
			n += jobs[i]->GetRayAmount();
		}
		CudaCheck(cudaDeviceSynchronize());

		if (n != 0){


			uint blockSize = 64;
			uint gridSize = (n + blockSize - 1) / blockSize;


			//execute engine
			cudaEvent_t start, stop;
			float time;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start, 0);

			EngineResultClear << <gridSize, blockSize >> >(n, jobs, jobsSize);

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			cudaEventDestroy(start);
			cudaEventDestroy(stop);

			std::cout << "RayClear Execution: " << time << "ms" << std::endl;
		}
		CudaCheck(cudaDeviceSynchronize());
	}

	cudaFree(jobs);
}
__host__ void ProcessJobs(std::vector<RayJob*>& hjobs, const Scene* scene){
	CudaCheck(cudaDeviceSynchronize());
	RayJob** jobs;
	uint jobsSize = hjobs.size();
	cudaMallocManaged((void**)&jobs, jobsSize*sizeof(RayJob*));

	CudaCheck(cudaDeviceSynchronize());

	for (int i = 0; i < jobsSize; i++){
		jobs[i] = hjobs[i];
	}

	CudaCheck(cudaDeviceSynchronize());

	if (jobsSize > 0){

		uint n = 0;
		uint samplesMax = 0;
		for (int i = 0; i < jobsSize; ++i){
			n += jobs[i]->GetRayAmount()* jobs[i]->GetSampleAmount();
			if (jobs[i]->GetSampleAmount()>samplesMax){
				samplesMax = jobs[i]->GetSampleAmount();
			}
		}

		CudaCheck(cudaDeviceSynchronize());

		if (n != 0){
			if (n > numRaysAllocated){
				//deviceRays.resize(n);

				CudaCheck(cudaFree(deviceRays));
				CudaCheck(cudaFree(deviceRaysB));

				CudaCheck(cudaMallocManaged((void**)&deviceRays, n*sizeof(Ray)));
				CudaCheck(cudaMallocManaged((void**)&deviceRaysB, n*sizeof(Ray)));
				numRaysAllocated = n;
			}


			//execute engine

			thrust::device_ptr<Ray> rayPtr = thrust::device_pointer_cast(deviceRays);


			cudaEvent_t start, stop;
			float time;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start, 0);


			int blockSize = 64;
			int gridSize = (n + blockSize - 1) / blockSize;

			RaySetup << <gridSize, blockSize >> >(n, jobs, jobsSize, deviceRays, WangHash(++raySeedGl), scene);


			CudaCheck(cudaDeviceSynchronize());
			uint numActive = n;

			int minGridSize;

			int* counter;
			CudaCheck(cudaMallocManaged((void**)&counter, sizeof(int)));
			counter[0] = 0;

			//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, EngineExecute, 0, 0);

			CudaCheck(cudaDeviceSynchronize());

			dim3 blockSizeE(WARP_SIZE, BLOCK_HEIGHT, 1);
			int blockWarps = (blockSizeE.x * blockSizeE.y + (WARP_SIZE - 1)) / WARP_SIZE;
			int numBlocks = (GPU_CORES + blockWarps - 1) / blockWarps;

			// Launch.



			//return kernel.launchTimed(numBlocks * blockSizeE.x * blockSizeE.y, blockSizeE);

			int* hitAtomic;
			CudaCheck(cudaMallocManaged((void**)&hitAtomic, sizeof(int)));
			*hitAtomic = 0;

			EngineExecute << <numBlocks * blockSizeE.x * blockSizeE.y, blockSizeE >> >(numActive, jobs, jobsSize, deviceRays, WangHash(++raySeedGl), scene, counter);
			CudaCheck(cudaDeviceSynchronize());

			CollectHits << <gridSize, blockSize >> >(numActive, jobs, jobsSize, deviceRays, deviceRaysB, WangHash(++raySeedGl), scene, hitAtomic);

			CudaCheck(cudaDeviceSynchronize());

			swap(deviceRays, deviceRaysB);

			for (uint i = 0; i < rayDepth - 1; ++i){
				CudaCheck(cudaDeviceSynchronize());

				size_t mem_tot_0 = 0;
				size_t mem_free_0 = 0;
				cudaMemGetInfo(&mem_free_0, &mem_tot_0);
				std::cout << "GPU Memory left: " << mem_free_0 / 1000000000.0f << " GB" << std::endl;

				CudaCheck(cudaDeviceSynchronize());

				numActive = hitAtomic[0];



				hitAtomic[0] = 0;

				blockSize = 64;
				gridSize = (numActive + blockSize - 1) / blockSize;
				CudaCheck(cudaDeviceSynchronize());

				counter[0] = 0;
				CudaCheck(cudaDeviceSynchronize());
				EngineExecute << <numBlocks * blockSizeE.x * blockSizeE.y, blockSizeE >> >(numActive, jobs, jobsSize, deviceRays, WangHash(++raySeedGl), scene, counter);
				CudaCheck(cudaDeviceSynchronize());
				CollectHits << <gridSize, blockSize >> >(numActive, jobs, jobsSize, deviceRays, deviceRaysB, WangHash(++raySeedGl), scene, hitAtomic);
				CudaCheck(cudaDeviceSynchronize());
				swap(deviceRays, deviceRaysB);

			}


			CudaCheck(cudaDeviceSynchronize());

			cudaFree(counter);
			cudaFree(hitAtomic);

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			cudaEventDestroy(start);
			cudaEventDestroy(stop);

			std::cout << "RayEngine Execution: " << time << "ms" << std::endl;

			CudaCheck(cudaDeviceSynchronize());
		}
	}

	cudaFree(jobs);

}

__host__ void Cleanup(){

	// empty the vector
	//deviceRays.clear();

	// deallocate any capacity which may currently be associated with vec
	//deviceRays.shrink_to_fit();
}

//__host__ void ClearResults(std::vector<RayJob*>& jobs){
//	internals::ClearResults(jobs);
//}
//__host__ void ProcessJobs(std::vector<RayJob*>& jobs, const Scene* scene){
//	internals::ProcessJobs(jobs, scene);
//}
//
//__host__ void Cleanup(){
//	internals::Cleanup();
//}
