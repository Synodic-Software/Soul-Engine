#include "Scene.cuh"

#define RAY_BIAS_DISTANCE 0.0002f 
#define BVH_STACK_SIZE 64
#define DYNAMIC_FETCH_THRESHOLD 20          // If fewer than this active, fetch new rays

#include "Utility\CUDA\CUDAHelper.cuh"
#include "Utility\Logger.h"

#include "Algorithms\Morton Code\MortonCode.h"

#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/functional.h>

Scene::Scene()
{
	objectsSize = 0;
	allocatedObjects = 0;
	allocatedSize = 0;

	compiledSize = 0;
	newFaceAmount = 0;

	objectListHost.clear();
	objectsToRemove.clear();
	objectListDevice = nullptr;
	mortonCodes = nullptr;
	faceIds = nullptr;
	objIds = nullptr;
	objectBitSetup = nullptr;
	//give the addresses for the data
	bvhHost = new BVH(&faceIds, &mortonCodes);
	CudaCheck(cudaMalloc((void **)&bvhDevice, sizeof(BVH)));

	Sky* skyHost = new Sky("Starmap.png");
	CudaCheck(cudaMalloc((void **)&sky, sizeof(Sky)));
	CudaCheck(cudaMemcpy(sky, skyHost, sizeof(Sky), cudaMemcpyHostToDevice));

	CudaCheck(cudaMalloc((void **)&sceneBoxDevice, sizeof(BoundingBox)));
}


Scene::~Scene()
{

	CudaCheck(cudaFree(objectBitSetup)); // hold a true for the first indice of each object
	CudaCheck(cudaFree(objIds)); //points to the object
	CudaCheck(cudaFree(faceIds));
	CudaCheck(cudaFree(mortonCodes));

	//Variables concerning object storage

	CudaCheck(cudaFree(objectListDevice));
	CudaCheck(cudaFree(objectRemoval));

	delete bvhHost;

	CudaCheck(cudaFree(bvhDevice));
	CudaCheck(cudaFree(sky));

}


struct is_scheduled
{
	__host__ __device__
		bool operator()(const Object* x)
	{
		return (x->requestRemoval);
	}
};


__global__ void FillBool(const uint n, bool* jobs, bool* fjobs, Face** faces, uint* objIds, Object** objects) {


	uint index = getGlobalIdx_1D_1D();


	if (index < n) {
		if (objects[objIds[index] - 1]->requestRemoval) {
			jobs[index] = true;
		}
		else {
			jobs[index] = false;
		}
		if (faces[index]->objectPointer->requestRemoval) {
			fjobs[index] = true;
		}
		else {
			fjobs[index] = false;
		}
	}
}

__global__ void GetFace(const uint n, uint* objIds, Object** objects, Face** faces, const uint offset) {


	uint index = getGlobalIdx_1D_1D();


	if (index >= n) {
		return;
	}

 	Object* obj = objects[objIds[offset + index] - 1];
	faces[offset + index] = obj->faces + (index - obj->localSceneIndex);
	faces[offset + index]->objectPointer = obj;

}


//add all new objects into the scene's arrays
__host__ bool Scene::Compile() {

	uint amountToRemove = objectsToRemove.size();

	if (newFaceAmount > 0 || amountToRemove > 0) {

		//bitSetup only has the first element of each object flagged
		//this extends that length and copies the previous results as well

		uint newSize = compiledSize + newFaceAmount;
		uint indicesToRemove = 0;
		uint removedOffset = 0;
		if (amountToRemove > 0) {

			bool* markers;
			bool* faceMarkers;
			CudaCheck(cudaMalloc((void**)&markers, compiledSize * sizeof(bool)));
			CudaCheck(cudaMalloc((void**)&faceMarkers, compiledSize * sizeof(bool)));

			thrust::device_ptr<bool> tempPtr = thrust::device_pointer_cast(markers);
			thrust::device_ptr<bool> faceTempPtr = thrust::device_pointer_cast(faceMarkers);


			//variables from the scene to the kernal

			uint blockSize = 64;
			uint gridSize = (compiledSize + blockSize - 1) / blockSize;

			//fill the mask with 1s or 0s
			FillBool << <gridSize, blockSize >> > (compiledSize, markers, faceMarkers, faceIds, objIds, objectListDevice);
			CudaCheck(cudaPeekAtLastError());
			CudaCheck(cudaDeviceSynchronize());


			//remove the requested
			thrust::device_ptr<bool> bitPtr = thrust::device_pointer_cast(objectBitSetup);

			thrust::device_ptr<bool> newEnd = thrust::remove_if(bitPtr, bitPtr + compiledSize, tempPtr, thrust::identity<bool>());
			CudaCheck(cudaDeviceSynchronize());

			indicesToRemove = bitPtr + compiledSize - newEnd;
			newSize = newSize - indicesToRemove;
			//objpointers
			thrust::device_ptr<uint> objPtr = thrust::device_pointer_cast(objIds);

			thrust::remove_if(objPtr, objPtr + compiledSize, tempPtr, thrust::identity<bool>());
			CudaCheck(cudaDeviceSynchronize());

			//faces
			thrust::device_ptr<Face*> facePtr = thrust::device_pointer_cast(faceIds);

			thrust::remove_if(facePtr, facePtr + compiledSize, faceTempPtr, thrust::identity<bool>());
			CudaCheck(cudaDeviceSynchronize());

			//actual object list
			thrust::device_ptr<Object*> objectsPtr = thrust::device_pointer_cast(objectListDevice);

			thrust::remove_if(objectsPtr, objectsPtr + objectsSize, is_scheduled());

			CudaCheck(cudaDeviceSynchronize());
			CudaCheck(cudaFree(markers));
			CudaCheck(cudaFree(faceMarkers));
		}

		if (newFaceAmount > 0) {

			if (allocatedSize < newSize) {
				Face** faceTemp;
				uint* objTemp;
				bool* objectBitSetupTemp;

				allocatedSize = glm::max(uint(allocatedSize * 1.5f), newSize);

				CudaCheck(cudaMalloc((void**)&faceTemp, allocatedSize * sizeof(Face*)));
				CudaCheck(cudaMalloc((void**)&objTemp, allocatedSize * sizeof(uint)));
				CudaCheck(cudaMalloc((void**)&objectBitSetupTemp, allocatedSize * sizeof(bool)));

				if (mortonCodes) {
					CudaCheck(cudaFree(mortonCodes));
				}
				CudaCheck(cudaMalloc((void**)&mortonCodes, allocatedSize * sizeof(uint64)));

				if (faceIds) {
					CudaCheck(cudaMemcpy(faceTemp, faceIds, compiledSize * sizeof(Face*), cudaMemcpyDeviceToDevice));
					CudaCheck(cudaFree(faceIds));
				}
				faceIds = faceTemp;

				if (objIds) {
					CudaCheck(cudaMemcpy(objTemp, objIds, compiledSize * sizeof(uint), cudaMemcpyDeviceToDevice));
					CudaCheck(cudaFree(objIds));
				}
				objIds = objTemp;

				if (objectBitSetup) {
					CudaCheck(cudaMemcpy(objectBitSetupTemp, objectBitSetup, compiledSize * sizeof(bool), cudaMemcpyDeviceToDevice));
					CudaCheck(cudaFree(objectBitSetup));
				}
				objectBitSetup = objectBitSetupTemp;

			}

			CudaCheck(cudaDeviceSynchronize());
			removedOffset = compiledSize - indicesToRemove;
			//for each new object, (all at the end of the array) fill with falses.
			thrust::device_ptr<bool> bitPtr = thrust::device_pointer_cast(objectBitSetup);
			thrust::fill(bitPtr + removedOffset, bitPtr + newSize, (bool)false);

			CudaCheck(cudaDeviceSynchronize());

		}


		CudaCheck(cudaDeviceSynchronize());

		//flag the first and setup state of life (only time iteration through objects should be done)
		uint l = 0;
		for (uint i = 0; i < objectsSize; i++) {
			if (!objectListHost[i]->ready) {
				CudaCheck(cudaMemset(objectBitSetup + l, true, sizeof(bool)));
				objectListHost[i]->ready = true;
			}
			
			objectListHost[i]->localSceneIndex = l;

			Object** objHolderHost = new Object*[1];

			Object* objDevice;
			CudaCheck(cudaMalloc((void**)&objDevice, sizeof(Object)));
			CudaCheck(cudaMemcpy(objDevice, objectListHost[i], sizeof(Object), cudaMemcpyHostToDevice));

			objHolderHost[0] = objDevice;
			CudaCheck(cudaMemcpy(objectListDevice + i, &objHolderHost[0], sizeof(Object*), cudaMemcpyHostToDevice));
			l += objectListHost[i]->faceAmount;
		}

		if (newFaceAmount > 0) {

			thrust::device_ptr<bool> bitPtr = thrust::device_pointer_cast(objectBitSetup);
			thrust::device_ptr<uint> objPtr = thrust::device_pointer_cast(objIds);
			CudaCheck(cudaDeviceSynchronize());

			thrust::inclusive_scan(bitPtr, bitPtr + newSize, objPtr);
			CudaCheck(cudaDeviceSynchronize());


			uint blockSize = 64;
			uint gridSize = ((newSize - removedOffset) + blockSize - 1) / blockSize;

			GetFace << <gridSize, blockSize >> > (newSize - removedOffset, objIds, objectListDevice, faceIds, removedOffset);
			CudaCheck(cudaPeekAtLastError());
			CudaCheck(cudaDeviceSynchronize());

		}
		CudaCheck(cudaDeviceSynchronize());

		//change the indice count of the scene
		compiledSize = newSize;
		newFaceAmount = 0;
		objectsToRemove.clear();
		return true;

	}
	else {
		return false;
	}


}


__host__ void Scene::Build(float deltaTime) {

	bool b = Compile();

	//calculate the morton code for each triangle

	uint blockSize = 64;
	uint gridSize = (compiledSize + blockSize - 1) / blockSize;

	CudaCheck(cudaDeviceSynchronize());

	MortonCode::Compute << <gridSize, blockSize >> > (compiledSize, mortonCodes, faceIds, objectListDevice, sceneBoxDevice);


	CudaCheck(cudaPeekAtLastError());
	CudaCheck(cudaDeviceSynchronize());
	thrust::device_ptr<uint64_t> keys(mortonCodes);
	thrust::device_ptr<Face*> values(faceIds);


	CudaCheck(cudaDeviceSynchronize());

	cudaEvent_t start, stop;
	float time;
	CudaCheck(cudaEventCreate(&start));
	CudaCheck(cudaEventCreate(&stop));
	CudaCheck(cudaEventRecord(start, 0));

	CudaCheck(cudaDeviceSynchronize());

	thrust::sort_by_key(keys, keys + compiledSize, values);

	CudaCheck(cudaDeviceSynchronize());

	CudaCheck(cudaEventRecord(stop, 0));
	CudaCheck(cudaEventSynchronize(stop));
	CudaCheck(cudaEventElapsedTime(&time, start, stop));
	CudaCheck(cudaEventDestroy(start));
	CudaCheck(cudaEventDestroy(stop));

	S_LOG_TRACE("     Sorting Execution: ", time, "ms");

	bvhHost->Build(compiledSize);
	CudaCheck(cudaMemcpy(bvhDevice, bvhHost, sizeof(BVH), cudaMemcpyHostToDevice));


}

//object pointer is host
__host__ uint Scene::AddObject(Object*& obj) {

	//if the size of objects stores increases, double the available size pool;
	if (objectsSize == allocatedObjects) {

		Object** objectsTemp;
		allocatedObjects *= 2;

		if (allocatedObjects == 0) {
			allocatedObjects = 1;
		}

		CudaCheck(cudaMalloc((void**)&objectsTemp, allocatedObjects * sizeof(Object*)));

		if (objectListDevice) {
			CudaCheck(cudaMemcpy(objectsTemp, objectListDevice, objectsSize * sizeof(Object*), cudaMemcpyDeviceToDevice));
			CudaCheck(cudaFree(objectListDevice));
		}

		objectListDevice = objectsTemp;
	}


	//update the scene's bounding volume

	sceneBoxHost.max = glm::max(sceneBoxHost.max, obj->box.max);
	sceneBoxHost.min = glm::min(sceneBoxHost.min, obj->box.min);

	CudaCheck(cudaMemcpy(sceneBoxDevice, &sceneBoxHost, sizeof(BoundingBox), cudaMemcpyHostToDevice));

	objectListHost.push_back(obj);

	//add the reference as the new object and increase the object count by 1
	Object** objHolderHost = new Object*[1];

	Object* objDevice;
	CudaCheck(cudaMalloc((void**)&objDevice, sizeof(Object)));
	CudaCheck(cudaMemcpy(objDevice, obj, sizeof(Object), cudaMemcpyHostToDevice));

	objHolderHost[0] = objDevice;
	CudaCheck(cudaMemcpy(objectListDevice + objectsSize, &objHolderHost[0], sizeof(Object*), cudaMemcpyHostToDevice));
	objectsSize++;
	newFaceAmount += obj->faceAmount;
	delete objHolderHost;

	return 0;
}

__host__ bool Scene::RemoveObject(const uint& tag) {
	objectRemoval[tag] = true;
	return true;
}
