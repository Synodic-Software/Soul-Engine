#include "Ray Engine/CUDA/RayEngine.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "Utility\CUDA\CUDAHelper.cuh"

//cant have AABB defined and not define WOOP_TRI
//#define WOOP_TRI
//#define WOOP_AABB

#define STACK_SIZE 64
#define DYNAMIC_FETCH_THRESHOLD 20          // If fewer than this active, fetch new rays
#define RAY_BIAS_DISTANCE 0.0002f 
#define p (1.0f + 2e-23f)
#define m (1.0f - 2e-23f)

namespace RayEngineCUDA {

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

	__global__ void RandomSetup(uint n, curandState* randomState, uint raySeed) {


		uint index = getGlobalIdx_1D_1D();

		if (index >= n) {
			return;
		}

		curandState randState;
		curand_init(index, 0, 0, &randState);

		randomState[index] = randState;

	}

	__global__ void EngineSetup(uint n, RayJob* jobs, int jobSize) {


		const int index = getGlobalIdx_1D_1D();

		if (index >= n) {
			return;
		}

		const int startIndex = 0;

		int cur = 0;
		((glm::vec4*)jobs[cur].camera.film.results)[index - startIndex] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
		jobs[cur].camera.film.hits[index - startIndex] = 0;

	}

	__global__ void RaySetup(uint n, uint jobSize, RayJob* job, Ray* rays, int* nAtomic, curandState* randomState) {

		const uint index = getGlobalIdx_1D_1D();

		if (index >= n) {
			return;
		}

		const auto startIndex = 0;
		const auto cur = 0;

		auto samples = job[cur].samples;
		const uint sampleIndex = (index - startIndex) / glm::ceil(samples); //the index of the pixel / sample
		const auto localIndex = (index - startIndex) % static_cast<int>(glm::ceil(samples));

		curandState randState = randomState[index];

		if (localIndex + 1 <= samples || curand_uniform(&randState) < __fsub_rd(samples, glm::floor(samples))) {

			Ray ray;
			ray.storage = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
			ray.resultOffset = sampleIndex;
			glm::vec3 orig;
			glm::vec3 dir;
			job[cur].camera.GenerateRay(sampleIndex, orig, dir, randState);
			ray.origin = glm::vec4(orig, 0.0f);
			ray.direction = glm::vec4(dir, 4000000000000.0f);

			atomicAdd(job[cur].camera.film.hits + sampleIndex, 1);

			const auto val = FastAtomicAdd(nAtomic);
			rays[val] = ray;

		}

		randomState[index] = randState;

	}

	__host__ __device__ __inline__ glm::vec3 PositionAlongRay(const Ray& ray, const float& t) {
		return glm::vec3(ray.origin.x, ray.origin.y, ray.origin.z) + t * glm::vec3(ray.direction.x, ray.direction.y, ray.direction.z);
	}

	//woop
	__host__ __device__ __inline__ bool FindTriangleIntersect(const glm::vec3& triA, const glm::vec3& triB, const glm::vec3& triC,
		const glm::vec3& rayO, int kx, int ky, int kz, float Sx, float Sy, float Sz,
		float& t, const float& tMax, float& bary1, float& bary2)
	{
		const glm::vec3 A = triA - rayO;
		const glm::vec3 B = triB - rayO;
		const glm::vec3 C = triC - rayO;

		const float Ax = A[kx] - Sx * A[kz];
		const float Ay = A[ky] - Sy * A[kz];
		const float Bx = B[kx] - Sx * B[kz];
		const float By = B[ky] - Sy * B[kz];
		const float Cx = C[kx] - Sx * C[kz];
		const float Cy = C[ky] - Sy * C[kz];

		float U = Cx * By - Cy * Bx;
		float V = Ax * Cy - Ay * Cx;
		float W = Bx * Ay - By * Ax;

		if (U == 0.0f || V == 0.0f || W == 0.0f) {
			double CxBy = (double)Cx*(double)By;
			double CyBx = (double)Cy*(double)Bx;
			U = (float)(CxBy - CyBx);
			double AxCy = (double)Ax*(double)Cy;
			double AyCx = (double)Ay*(double)Cx;
			V = (float)(AxCy - AyCx);
			double BxAy = (double)Bx*(double)Ay;
			double ByAx = (double)By*(double)Ax;
			W = (float)(BxAy - ByAx);
		}

		if ((U < 0.0f || V < 0.0f || W < 0.0f) &&
			(U > 0.0f || V > 0.0f || W > 0.0f)) {
			return false;
		}

		float det = U + V + W;

		if (det == 0.0f) {
			return false;
		}

		const float Az = Sz * A[kz];
		const float Bz = Sz * B[kz];
		const float Cz = Sz * C[kz];
		const float T = U * Az + V * Bz + W * Cz;

		/*int det_sign = glm::sign(det);
		if (xorf(T, det_sign) < 0.0f) ||
			xorf(T, det_sign) > hit.t * xorf(det, det_sign)){
			return false;
		}*/

		const float rcpDet = 1.0f / det;

		bary1 = V * rcpDet;
		bary2 = W * rcpDet;
		t = T * rcpDet;

		return (t > EPSILON &&t < tMax);
	}


	//Moller-Trumbore
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


	__global__ void ProcessHits(uint n, RayJob* job, int jobSize, Ray* rays, Ray* raysNew, Sky* sky, Face* faces, Vertex* vertices, Material* materials, int * nAtomic, curandState* randomState) {

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

			col = glm::vec3(ray.storage.x, ray.storage.y, ray.storage.z)*sky->ExtractColour({ ray.direction.x, ray.direction.y, ray.direction.z });

		}
		else {

			Face face = faces[faceHit];

			Material mat = materials[face.material];

			glm::vec2 bary = ray.bary;

			glm::vec3 n0 = vertices[face.indices.x].position;
			glm::vec3 n1 = vertices[face.indices.y].position;
			glm::vec3 n2 = vertices[face.indices.z].position;

			glm::vec3 bestNormal = glm::normalize(glm::cross(n1 - n0, n2 - n0));

			/*glm::vec3 n0 = vertices[faceHit->indices.x].normal;
			glm::vec3 n1 = vertices[faceHit->indices.y].normal;
			glm::vec3 n2 = vertices[faceHit->indices.z].normal;

			glm::vec3 bestNormal = (1 - bary.x - bary.y) * n0 + bary.x * n1 + bary.y * n2;*/

			glm::vec2 uv0 = vertices[face.indices.x].textureCoord;
			glm::vec2 uv1 = vertices[face.indices.y].textureCoord;
			glm::vec2 uv2 = vertices[face.indices.z].textureCoord;

			glm::vec2 bestUV = (1.0f - bary.x - bary.y) * uv0 + bary.x * uv1 + bary.y * uv2;

			glm::vec3 orientedNormal = glm::dot(bestNormal, glm::vec3(ray.direction.x, ray.direction.y, ray.direction.z)) < 0 ? bestNormal : bestNormal * -1.0f;

			glm::vec3 biasVector = (RAY_BIAS_DISTANCE * orientedNormal);

			glm::vec3 bestIntersectionPoint = PositionAlongRay(ray, hitT);

			glm::vec3 accumulation;
			accumulation.x = ray.storage.x * mat.emit.x;
			accumulation.y = ray.storage.y * mat.emit.y;
			accumulation.z = ray.storage.z * mat.emit.z;

			//tex2D<float>(texObj, tu, tv)
			//unsigned char blue = tex2D<unsigned char>(mat->texObj, (4 * localIndex), localIndex);
			//unsigned char green = tex2D<unsigned char>(mat->texObj, (4 * localIndex) + 1, localIndex);
			//unsigned char red = tex2D<unsigned char>(mat->texObj, (4 * localIndex) + 2, localIndex);

			float4 PicCol = tex2DLod<float4>(mat.diffuseImage.texObj, bestUV.x, bestUV.y, 0.0f);
			//float PicCol = tex2D<float>(mat->texObj, bestUV.x * 50, bestUV.y * 50);
			ray.storage *= glm::vec4(PicCol.x, PicCol.y, PicCol.z, 1.0f);

			//ray.storage *= mat->diffuse;

			float r1 = 2 * PI * curand_uniform(&randState);
			float r2 = curand_uniform(&randState);
			float r2s = sqrtf(r2);

			glm::vec3 u = glm::normalize(glm::cross((glm::abs(orientedNormal.x) > .1 ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0)), orientedNormal));
			glm::vec3 v = glm::cross(orientedNormal, u);

			ray.origin = glm::vec4(bestIntersectionPoint + biasVector, 0.0f);
			ray.direction = glm::vec4(glm::normalize(u*cos(r1)*r2s + v * sin(r1)*r2s + orientedNormal * sqrtf(1 - r2)), 40000000000000.0f);

			col = accumulation;

			raysNew[FastAtomicAdd(nAtomic)] = ray;

			//update the state
			randomState[index] = randState;
		}

		int samples = job[cur].camera.film.hits[localIndex];

		col /= samples;

		glm::vec4* pt = &((glm::vec4*)job[cur].camera.film.results)[localIndex];

		atomicAdd(&(pt->x), col.x);

		atomicAdd(&(pt->y), col.y);

		atomicAdd(&(pt->z), col.z);


	}

	__inline__ __device__ float up(float a) { return a > 0.0f ? a * p : a * m; }
	__inline__ __device__ float dn(float a) { return a > 0.0f ? a * m : a * p; }

	__inline__ __device__ float Up(float a) { return a * p; }
	__inline__ __device__ glm::vec3 Up(glm::vec3 a) { return a * p; }
	__inline__ __device__ float Dn(float a) { return a * m; }
	__inline__ __device__ glm::vec3 Dn(glm::vec3 a) { return a * m; }

	__global__ void ExecuteJobs(uint n, Ray* rays, BVHData* bvhP, Vertex* vertices, Face* faces, int* counter) {

		Node * traversalStack[STACK_SIZE];
		traversalStack[0] = nullptr; // Bottom-most entry.

		float eps = exp2f(-80.0f);

		char    stackPtr;                       // Current position in traversal stack.
		Node*   currentLeaf;                       // First postponed leaf, non-negative if none.
		Node*   currentNode = nullptr;				// Non-negative: current internal node, negative: second postponed leaf.
		int     rayidx;

#if defined WOOP_AABB

		float idirxNear;
		float idiryNear;
		float idirzNear;
		float idirxFar;
		float idiryFar;
		float idirzFar;

#else

		float oodx;
		float oody;
		float oodz;

		float idirx;
		float idiry;
		float idirz;

#endif

#if defined WOOP_AABB || defined WOOP_TRI
		//ray precalc
		int kz;
		int kx;
		int ky;
#endif

#if defined WOOP_TRI
		float Sx;
		float Sy;
		float Sz;
#endif

		//scene data
		Ray ray;
		BVHData bvh = *bvhP;

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

				//ray local storage + precalculations
				ray = rays[rayidx];

#if defined WOOP_AABB || defined WOOP_TRI


				//triangle precalc
				glm::vec3 absDir = glm::abs(ray.direction);
				if (absDir.x >= absDir.y&&absDir.x >= absDir.z) {
					kz = 0;
				}
				else if (absDir.y >= absDir.x&&absDir.y >= absDir.z) {
					kz = 1;
				}
				else {
					kz = 2;
				}

				kx = kz + 1; if (kx == 3) kx = 0;
				ky = kx + 1; if (ky == 3) ky = 0;

				if (ray.direction[kz] < 0.0f) {
					swap(kx, ky);
				}
#endif

#if	defined WOOP_TRI
				Sx = ray.direction[kx] / ray.direction[kz];
				Sy = ray.direction[ky] / ray.direction[kz];
				Sz = 1.0f / ray.direction[kz];
#endif

#if defined WOOP_AABB

				glm::vec3 rdir = 1.0f / ray.direction;
				idirxNear = Dn(Dn(rdir[kx]));
				idiryNear = Dn(Dn(rdir[ky]));
				idirzNear = Dn(Dn(rdir[kz]));
				idirxFar = Up(Up(rdir[kx]));
				idiryFar = Up(Up(rdir[ky]));
				idirzFar = Up(Up(rdir[kz]));

#else

				idirx = 1.0f / (fabsf(ray.direction.x) > eps ? ray.direction.x : copysignf(eps, ray.direction.x));
				idiry = 1.0f / (fabsf(ray.direction.y) > eps ? ray.direction.y : copysignf(eps, ray.direction.y));
				idirz = 1.0f / (fabsf(ray.direction.z) > eps ? ray.direction.z : copysignf(eps, ray.direction.z));

				//non-moded
				oodx = ray.origin.x * idirx;
				oody = ray.origin.y * idiry;
				oodz = ray.origin.z * idirz;

#endif

				// Setup traversal.
				stackPtr = 0;
				currentLeaf = nullptr;   // No postponed leaf.
				currentNode = bvh.root;   // Start from the root.
				ray.currentHit = uint(-1);  // No triangle intersected so far.
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

#if defined WOOP_AABB

					//grab the modifyable bounds
					float nearX0 = b0Min[kx], farX0 = b0Max[kx];
					float nearY0 = b0Min[ky], farY0 = b0Max[ky];
					float nearZ0 = b0Min[kz], farZ0 = b0Max[kz];
					float nearX1 = b1Min[kx], farX1 = b1Max[kx];
					float nearY1 = b1Min[ky], farY1 = b1Max[ky];
					float nearZ1 = b1Min[kz], farZ1 = b1Max[kz];

					if (ray.direction[kx] < 0.0f) swap(nearX0, farX0);
					if (ray.direction[ky] < 0.0f) swap(nearY0, farY0);
					if (ray.direction[kz] < 0.0f) swap(nearZ0, farZ0);
					if (ray.direction[kx] < 0.0f) swap(nearX1, farX1);
					if (ray.direction[ky] < 0.0f) swap(nearY1, farY1);
					if (ray.direction[kz] < 0.0f) swap(nearZ1, farZ1);

					glm::vec3 lower0 = Dn(glm::abs(glm::vec3(ray.origin) - b0Min));
					glm::vec3 upper0 = Up(glm::abs(glm::vec3(ray.origin) - b0Max));
					glm::vec3 lower1 = Dn(abs(glm::vec3(ray.origin) - b1Min));
					glm::vec3 upper1 = Up(abs(glm::vec3(ray.origin) - b1Max));

					float max_z0 = glm::max(lower0[kz], upper0[kz]);
					float max_z1 = glm::max(lower1[kz], upper1[kz]);

					//calc the rror and update the origin
					float err_near_x0 = Up(lower0[kx] + max_z0);
					float err_near_y0 = Up(lower0[ky] + max_z0);
					float err_near_x1 = Up(lower1[kx] + max_z1);
					float err_near_y1 = Up(lower1[ky] + max_z1);

					float oodxNear0 = up(ray.origin[kx] + Up(eps*err_near_x0));
					float oodyNear0 = up(ray.origin[ky] + Up(eps*err_near_y0));
					float oodzNear0 = ray.origin[kz];
					float oodxNear1 = up(ray.origin[kx] + Up(eps*err_near_x1));
					float oodyNear1 = up(ray.origin[ky] + Up(eps*err_near_y1));
					float oodzNear1 = ray.origin[kz];

					float err_far_x0 = Up(upper0[kx] + max_z0);
					float err_far_y0 = Up(upper0[ky] + max_z0);
					float err_far_x1 = Up(upper1[kx] + max_z1);
					float err_far_y1 = Up(upper1[ky] + max_z1);

					float oodxFar0 = dn(ray.origin[kx] - Up(eps*err_far_x0));
					float oodyFar0 = dn(ray.origin[ky] - Up(eps*err_far_y0));
					float oodzFar0 = ray.origin[kz];
					float oodxFar1 = dn(ray.origin[kx] - Up(eps*err_far_x1));
					float oodyFar1 = dn(ray.origin[ky] - Up(eps*err_far_y1));
					float oodzFar1 = ray.origin[kz];

					if (ray.direction[kx] < 0.0f) swap(oodxNear0, oodxFar0);
					if (ray.direction[ky] < 0.0f) swap(oodyNear0, oodyFar0);
					if (ray.direction[kx] < 0.0f) swap(oodxNear1, oodxFar1);
					if (ray.direction[ky] < 0.0f) swap(oodyNear1, oodyFar1);



					// Intersect the ray against the child nodes.
					const float c0lox = (nearX0 - oodxNear0) * idirxNear;
					const float c0hix = (farX0 - oodxFar0) * idirxFar;
					const float c0loy = (nearY0 - oodyNear0) * idiryNear;
					const float c0hiy = (farY0 - oodyFar0) * idiryFar;
					const float c0loz = (nearZ0 - oodzNear0)   * idirzNear;
					const float c0hiz = (farZ0 - oodzFar0)   * idirzFar;
					const float c1loz = (nearZ1 - oodzNear1)   * idirzNear;
					const float c1hiz = (farZ1 - oodzFar1)   * idirzFar;
					const float c0min = spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, ray.origin.w);
					const float c0max = spanEndKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, ray.direction.w);

					const float c1lox = (nearX1 - oodxNear1) * idirxNear;
					const float c1hix = (farX1 - oodxFar1) * idirxFar;
					const float c1loy = (nearY1 - oodyNear1) * idiryNear;
					const float c1hiy = (farY1 - oodyFar1) * idiryFar;
					const float c1min = spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, ray.origin.w);
					const float c1max = spanEndKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, ray.direction.w);

#else

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

#endif

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
					const uint faceID = currentLeaf->faceID;
					const glm::uvec3 face = faces[faceID].indices;

					float bary1;
					float bary2;
					float tTemp;

#if defined	WOOP_TRI

					const bool test = FindTriangleIntersect(vertices[face.x].position, vertices[face.y].position, vertices[face.z].position,
						ray.origin, kx, ky, kz, Sx, Sy, Sz,
#else
					const bool test = FindTriangleIntersect(vertices[face.x].position, vertices[face.y].position, vertices[face.z].position,
						ray.origin, ray.direction, { idirx, idiry, idirz },
#endif
						tTemp, ray.direction.w, bary1, bary2);

					if (test) {

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



	__global__ void Test(int* nAtomic) {

		uint index = getGlobalIdx_1D_1D();

		if (index >= 500) {
			return;
		}

		const auto val = FastAtomicAdd(nAtomic);
	}

	void LaunchTest(int* nAtomic) {

		const auto blockSize = 64;
		Test << <(500 + blockSize - 1) / blockSize, blockSize >> > (nAtomic);
		CudaCheck(cudaPeekAtLastError());
		CudaCheck(cudaDeviceSynchronize());

	}

}