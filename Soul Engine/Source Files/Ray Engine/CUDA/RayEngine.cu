#include "Ray Engine\RayEngine.cuh"

uint raySeed=0;

__device__ uint getGlobalIdx_1D_1D()
{
	return blockIdx.x *blockDim.x + threadIdx.x;
}

__global__ void EngineExecute(uint n, RayJob& jobs, uint raySeed){
	uint index = getGlobalIdx_1D_1D();

	if (index < n){
		RayJob job = jobs;
		uint n = 0;

		while (jobs.nextRay != NULL && !(index < n + jobs.rayAmount)){
			n += job.rayAmount*job.samples;
			job = *job.nextRay;
		}

		uint localIndex = index - n;

		Ray ray = job.camera->SetupRay(index, n, raySeed);

		glm::vec2 fov= job.camera->FieldOfView();
		float aspectRatio = fov.x / fov.y;
		int screenX = (n*aspectRatio) / (aspectRatio + 1);
		int screenY = n / screenX;
		uint i = localIndex % screenX;
		uint j = localIndex / screenX;

		//calculate something


		if (job.type != RayOBJECT_ID&&!RayCOLOUR_TO_TEXTURE){
			job.GetResultFloat()[localIndex] = glm::vec3(0.5f, 0.5f, 0.5f);
		}
		else if (RayCOLOUR_TO_TEXTURE){
			job.GetResultBuffer()[localIndex] = glm::vec4(0.5f, 0.5f, 0.5f,1.0f);
		}
		else{
			job.GetResultInt()[localIndex] = 1;
		}
	}
}

__host__ void ProcessJobs(RayJob* jobs){
	raySeed++;

	if (jobs!=NULL){
	uint n = 0;

	RayJob* temp = jobs;
	n += temp->rayAmount;
	while (temp->nextRay != NULL){
		temp = temp->nextRay;
		n += temp->rayAmount*temp->samples;
	}

	if (n!=0){

		const int warpSize = 32;
		const int maxGridSize = 112; // this is 8 blocks per MP for a Telsa C2050

		int warpCount = (n / warpSize) + (((n % warpSize) == 0) ? 0 : 1);
		int warpPerBlock = glm::max(1, glm::min(4, warpCount));

		// For the cdiv kernel, the block size is allowed to grow to
		// four warps per block, and the block count becomes the warp count over four
		// or the GPU "fill" whichever is smaller
		int threadCount = warpSize * warpPerBlock;
		int blockCount = glm::min(maxGridSize, glm::max(1, warpCount / warpPerBlock));
		dim3 BlockDim = dim3(threadCount, 1, 1);
		dim3 GridDim = dim3(blockCount, 1, 1);


		//execute engine
		EngineExecute << <GridDim, BlockDim >> >(n, *jobs, raySeed);

	}
	}


}