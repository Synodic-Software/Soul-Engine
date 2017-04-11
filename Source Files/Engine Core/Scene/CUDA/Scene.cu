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

	objects = nullptr;
	mortonCodes = nullptr;
	faces = nullptr;
	vertices = nullptr;
	materials = nullptr;
	tets = nullptr;

	faceAmount = 0;
	vertexAmount = 0;
	tetAmount = 0;
	materialAmount = 0;
	objectAmount = 0;

	faceAllocated = 0;
	vertexAllocated = 0;
	tetAllocated = 0;
	materialAllocated = 0;
	objectAllocated = 0;

	//
	bvhHost = new BVH();
	CudaCheck(cudaMalloc((void **)&bvh, sizeof(BVH)));

	Sky* skyHost = new Sky("Starmap.png");
	CudaCheck(cudaMalloc((void **)&sky, sizeof(Sky)));
	CudaCheck(cudaMemcpy(sky, skyHost, sizeof(Sky), cudaMemcpyHostToDevice));
}


Scene::~Scene()
{

	CudaCheck(cudaFree(mortonCodes));
	CudaCheck(cudaFree(faces));
	CudaCheck(cudaFree(vertices));
	CudaCheck(cudaFree(tets));
	CudaCheck(cudaFree(materials));
	CudaCheck(cudaFree(objects));

	delete bvhHost;

	CudaCheck(cudaFree(bvh));
	CudaCheck(cudaFree(sky));

}

__host__ void Scene::Build(float deltaTime) {

	Compile();

	//calculate the morton code for each triangle
	uint blockSize = 64;
	uint gridSize = (faceAmount + blockSize - 1) / blockSize;

	CudaCheck(cudaDeviceSynchronize());

	MortonCode::Compute << <gridSize, blockSize >> > (faceAmount, mortonCodes, faces, vertices, sceneBox);

	CudaCheck(cudaPeekAtLastError());
	CudaCheck(cudaDeviceSynchronize());

	thrust::device_ptr<uint64_t> keys(mortonCodes);
	thrust::device_ptr<Face> values(faces);

	CudaCheck(cudaDeviceSynchronize());

	cudaEvent_t start, stop;
	float time;
	CudaCheck(cudaEventCreate(&start));
	CudaCheck(cudaEventCreate(&stop));
	CudaCheck(cudaEventRecord(start, 0));

	CudaCheck(cudaDeviceSynchronize());

	thrust::sort_by_key(keys, keys + faceAmount, values);

	CudaCheck(cudaDeviceSynchronize());

	CudaCheck(cudaEventRecord(stop, 0));
	CudaCheck(cudaEventSynchronize(stop));
	CudaCheck(cudaEventElapsedTime(&time, start, stop));
	CudaCheck(cudaEventDestroy(start));
	CudaCheck(cudaEventDestroy(stop));

	S_LOG_TRACE("     Sorting Execution: ", time, "ms");

	/*bvhHost->Build(faceAmount, mortonCodes, faces, vertices);
	CudaCheck(cudaMemcpy(bvh, bvhHost, sizeof(BVH), cudaMemcpyHostToDevice));*/
}

void Scene::Compile() {

	if (addList.size() > 0) {

		uint faceAmountPrevious = faceAmount;
		uint vertexAmountPrevious = vertexAmount;
		uint tetAmountPrevious = tetAmount;
		uint materialAmountPrevious = materialAmount;
		uint objectAmountPrevious = objectAmount;

		for (int i = 0; i < addList.size(); ++i) {

			tetAmount += addList[i]->tetAmount;
			faceAmount += addList[i]->faceAmount;
			vertexAmount += addList[i]->verticeAmount;
			materialAmount += addList[i]->materialAmount;
			++objectAmount;

		}



		//vertex resize
		if (vertexAmount > vertexAllocated) {
			Vertex* vertexTemp;

			vertexAllocated = glm::max(vertexAmount, uint(vertexAllocated*1.5f));
			CudaCheck(cudaMalloc((void**)&vertexTemp, vertexAllocated * sizeof(Vertex)));

			if (vertices) {
				CudaCheck(cudaMemcpy(vertexTemp, vertices, vertexAmountPrevious * sizeof(Vertex), cudaMemcpyDeviceToDevice));
				CudaCheck(cudaFree(vertices));
			}

			vertices = vertexTemp;
		}

		//face resize + morton codes
		if (faceAmount > faceAllocated) {
			Face* facesTemp;

			faceAllocated = glm::max(faceAmount, uint(faceAllocated*1.5f));
			CudaCheck(cudaMalloc((void**)&facesTemp, faceAllocated * sizeof(Face)));

			if (faces) {
				CudaCheck(cudaMemcpy(facesTemp, faces, faceAmountPrevious * sizeof(Face), cudaMemcpyDeviceToDevice));
				CudaCheck(cudaFree(faces));
			}

			faces = facesTemp;

			if (mortonCodes) {
				CudaCheck(cudaFree(mortonCodes));
			}
			CudaCheck(cudaMalloc((void**)&mortonCodes, faceAllocated * sizeof(uint64)));

		}

		//tet resize
		if (tetAmount > tetAllocated) {
			Tet* tetsTemp;

			tetAllocated = glm::max(tetAmount, uint(tetAllocated*1.5f));
			CudaCheck(cudaMalloc((void**)&tetsTemp, tetAllocated * sizeof(Tet)));

			if (tets) {
				CudaCheck(cudaMemcpy(tetsTemp, tets, tetAmountPrevious * sizeof(Tet), cudaMemcpyDeviceToDevice));
				CudaCheck(cudaFree(tets));
			}

			tets = tetsTemp;
		}

		//material resize
		if (materialAmount > materialAllocated) {
			Material* materialsTemp;

			materialAllocated = glm::max(materialAmount, uint(materialAllocated*1.5f));
			CudaCheck(cudaMalloc((void**)&materialsTemp, materialAllocated * sizeof(Material)));

			if (materials) {
				CudaCheck(cudaMemcpy(materialsTemp, materials, tetAmountPrevious * sizeof(Material), cudaMemcpyDeviceToDevice));
				CudaCheck(cudaFree(materials));
			}

			materials = materialsTemp;
		}

		//object resize
		if (objectAmount > objectAllocated) {
			MiniObject* objectsTemp;

			objectAllocated = glm::max(objectAmount, uint(objectAllocated*1.5f));
			CudaCheck(cudaMalloc((void**)&objectsTemp, objectAllocated * sizeof(MiniObject)));

			if (objects) {
				CudaCheck(cudaMemcpy(objectsTemp, objects, objectAmountPrevious * sizeof(MiniObject), cudaMemcpyDeviceToDevice));
				CudaCheck(cudaFree(objects));
			}

			objects = objectsTemp;
		}

		uint faceOffset = faceAmountPrevious;
		uint vertexOffset = vertexAmountPrevious;
		uint tetOffset = tetAmountPrevious;
		uint materialOffset = materialAmountPrevious;
		uint objectOffset = objectAmountPrevious;

		for (int i = 0; i < addList.size(); ++i) {

			//update the scene's bounding volume
			sceneBox.max = glm::max(sceneBox.max, addList[i]->box.max);
			sceneBox.min = glm::min(sceneBox.min, addList[i]->box.min);

			//create the modified host data to upload
			std::vector<Vertex> tempVertices(addList[i]->verticeAmount);
			std::vector<Face> tempFaces(addList[i]->faceAmount);
			std::vector<Tet> tempTets(addList[i]->tetAmount);
			std::vector<Material> tempMaterials(addList[i]->materialAmount);

			//create the minified object from the input object
			MiniObject tempObject(*addList[i]);

			uint maxIter = glm::max(addList[i]->materialAmount, glm::max(addList[i]->verticeAmount, glm::max(addList[i]->faceAmount, addList[i]->tetAmount)));

			for (uint t = 0; t < maxIter; ++t) {
				if (t < addList[i]->verticeAmount) {
					tempVertices[t] = addList[i]->vertices[t];
					tempVertices[t].object = objectOffset;
				}
				if (t < addList[i]->faceAmount) {
					tempFaces[t] = addList[i]->faces[t];
					tempFaces[t].material += materialOffset;
				}
				if (t < addList[i]->tetAmount) {
					tempTets[t] = addList[i]->tets[t];
					tempTets[t].material += materialOffset;
					tempTets[t].object = objectOffset;
				}
				if (t < addList[i]->materialAmount) {
					tempMaterials[t] = addList[i]->materials[t];
				}
			}


			//upload the data
			CudaCheck(cudaMemcpy(vertices + vertexOffset, tempVertices.data(), tempVertices.size() * sizeof(Vertex), cudaMemcpyHostToDevice));
			CudaCheck(cudaMemcpy(faces + faceOffset, tempFaces.data(), tempFaces.size() * sizeof(Face), cudaMemcpyHostToDevice));
			CudaCheck(cudaMemcpy(tets + tetOffset, tempTets.data(), tempTets.size() * sizeof(Tet), cudaMemcpyHostToDevice));
			CudaCheck(cudaMemcpy(materials + materialOffset, tempMaterials.data(), tempMaterials.size() * sizeof(Material), cudaMemcpyHostToDevice));
			CudaCheck(cudaMemcpy(objects + objectOffset, &tempObject, sizeof(MiniObject), cudaMemcpyHostToDevice));




			//update the offsets
			tetOffset += addList[i]->tetAmount;
			faceOffset += addList[i]->faceAmount;
			vertexOffset += addList[i]->verticeAmount;
			materialOffset += addList[i]->materialAmount;
			++objectOffset;

		}

		//clear the list
		addList.clear();
	}
}

//object pointer is host
void Scene::AddObject(Object* obj) {
	addList.push_back(obj);
}

void Scene::RemoveObject(Object* obj) {
	removeList.push_back(obj);
	S_LOG_WARNING("Removal of objects from a scene not yet implemented");
}
