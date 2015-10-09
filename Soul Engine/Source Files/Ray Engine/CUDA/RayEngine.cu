#include "Ray Engine\RayEngine.cuh"

uint raySeedGl=0;

inline __device__ int getGlobalIdx_1D_1D()
{
	return blockIdx.x *blockDim.x + threadIdx.x;
}

__global__ void EngineExecute(uint n, RayJob& jobs, uint raySeed){
	uint index = getGlobalIdx_1D_1D();


	thrust::default_random_engine rng(randHash(raySeed) * randHash(index) * randHash(raySeed));
	thrust::uniform_real_distribution<float> uniformDistribution(0.0f, 1.0f); // Changed to 0.0 and 1.0 so I could reuse it for aperture sampling below.

	if (index < n){
		RayJob job = jobs;
		uint startIndex = 0;

		while (jobs.nextRay != NULL && !(index < startIndex + jobs.rayAmount)){
			startIndex += job.rayAmount*job.samples;
			job = *job.nextRay;
		}

		uint localIndex = index - startIndex;

		Ray ray = job.camera->SetupRay(localIndex, job.rayAmount, rng);

		glm::vec2 fov= job.camera->FieldOfView();
		float aspectRatio = fov.x / fov.y;
		glm::uvec2 screen = job.camera->GetResolution();
		uint i = localIndex / screen.x;
		uint j = localIndex % screen.y;

		//calculate something


		if (job.type != RayOBJECT_ID&&job.type != RayCOLOUR_TO_BUFFER){
			job.resultsF[localIndex] = glm::vec3(0.5f, 0.5f, 0.5f);
		}
		else if (job.type==RayCOLOUR_TO_BUFFER){
			
			float jitterValueX = uniformDistribution(rng);
			job.resultsT[localIndex] = make_float4(jitterValueX, jitterValueX, jitterValueX, 1.0f);
		}
		else if (job.resultsI!=NULL){
			job.resultsI[localIndex] = 1;
		}
	}
}

__host__ void ProcessJobs(RayJob* jobs){
	raySeedGl++;

	if (jobs!=NULL){
	uint n = 0;

	RayJob* temp = jobs;
	n += temp->rayAmount;
	while (temp->nextRay != NULL){
		temp = temp->nextRay;
		n += temp->rayAmount*temp->samples;
	}

	if (n!=0){

		int blockSize;   // The launch configurator returned block size 
		int minGridSize; // The minimum grid size needed to achieve the 
		// maximum occupancy for a full device launch 
		int gridSize;    // The actual grid size needed, based on input size 

		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
			EngineExecute, 0, n);
		// Round up according to array size 
		gridSize = (n + blockSize - 1) / blockSize;


		//execute engine
		EngineExecute << <gridSize, blockSize >> >(n, *jobs, raySeedGl);
		CudaCheck(cudaDeviceSynchronize());
	}
	}


}