#include "Scene.cuh"

#define RAY_BIAS_DISTANCE 0.0002f 
#define BVH_STACK_SIZE 64
#define DYNAMIC_FETCH_THRESHOLD 20          // If fewer than this active, fetch new rays

#include "Utility\CUDA\CUDAHelper.cuh"
#include "Utility\Logger.h"

#include "Algorithms\Morton Code\MortonCode.h"

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

Scene::Scene()
{

	objects = nullptr;
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
	CudaCheck(cudaMalloc((void **)&bvhData, sizeof(BVHData)));

	skyHost = new Sky("Starmap.png");
	CudaCheck(cudaMalloc((void **)&sky, sizeof(Sky)));
	CudaCheck(cudaMemcpy(sky, skyHost, sizeof(Sky), cudaMemcpyHostToDevice));
}


Scene::~Scene()
{

	CudaCheck(cudaFree(faces));
	CudaCheck(cudaFree(vertices));
	CudaCheck(cudaFree(tets));
	CudaCheck(cudaFree(materials));
	CudaCheck(cudaFree(objects));

	delete skyHost;

	CudaCheck(cudaFree(bvhData));
	CudaCheck(cudaFree(sky));

}

struct depthAt
{
	int depth;
	Vertex* vertices;
	MiniObject* objects;

	depthAt(int d, Vertex* vert, MiniObject* obj) :
		depth(d), vertices(vert), objects(obj) {};

	__host__ __device__
		bool operator()(const Face& x)
	{
		return objects[vertices[x.indices.x].object].tSize >= depth;
	}
};

__global__ void GrabMortons(const uint n, const uint depth, uint64* mortons, uint* indices, Face* faces, Vertex* vertices, MiniObject* objects) {

	uint index = getGlobalIdx_1D_1D();

	if (index >= n) {
		return;
	}

	Face face = faces[index];
	MiniObject obj = objects[vertices[face.indices.x].object];

	uint64 mort;
	if (depth > obj.tSize - 1) {
		mort = face.mortonCode;
	}
	else {
		mort = obj.transforms[depth].morton;
	}

	mortons[index] = mort;
}

__global__ void ScatterFaces(const uint n, uint64* mortons, uint* indices, Face* facesOld,Face* faces) {

	uint index = getGlobalIdx_1D_1D();

	if (index >= n) {
		return;
	}

	faces[indices[index]] = facesOld[index];
}

__host__ void Scene::Build(float deltaTime) {

	Compile();

	if (faceAmount > 0) {
		//calculate the morton code for each triangle
		uint blockSize = 64;
		uint gridSize = (faceAmount + blockSize - 1) / blockSize;

		MortonCode::Compute << <gridSize, blockSize >> > (faceAmount, faces, vertices);
		CudaCheck(cudaPeekAtLastError());
		CudaCheck(cudaDeviceSynchronize());

		//calc the depth to sort
		int depth = 0;
		for (int i = 0; i < addList.size(); ++i) {
			int size = addList[i].first.size();
			if (size > depth) {
				depth = size;
			}
		}

		for (int i = 0; i < cameraList.size(); ++i) {
			int size = cameraList[i].first.size();
			if (size > depth) {
				depth = size;
			}
		}

		for (int depthToSort = depth; depthToSort >= 0; --depthToSort) {

			thrust::counting_iterator<uint> first(0);
			thrust::counting_iterator<uint> last = first + faceAmount;
			thrust::device_vector<uint> resultsIndices(faceAmount);
			thrust::device_vector<Face> resultsFaces(faceAmount);

			auto resultIndex = thrust::copy_if(thrust::device, first, last, faces, resultsIndices.begin(), depthAt(depthToSort, vertices, objects));
			auto resultFace = thrust::copy_if(thrust::device, faces, faces + faceAmount, resultsFaces.begin(), depthAt(depthToSort, vertices, objects));

			uint size = resultIndex - resultsIndices.begin();
			thrust::device_vector<uint64> mortonsToSort(size);

			blockSize = 64;
			gridSize = (size + blockSize - 1) / blockSize;

			GrabMortons << <gridSize, blockSize >> > (size, depthToSort, thrust::raw_pointer_cast(&mortonsToSort[0]), thrust::raw_pointer_cast(&resultsIndices[0]), thrust::raw_pointer_cast(&resultsFaces[0]), vertices, objects);
			CudaCheck(cudaPeekAtLastError());
			CudaCheck(cudaDeviceSynchronize());

			thrust::sort_by_key(thrust::device, mortonsToSort.begin(), mortonsToSort.end(), resultsFaces.begin());

			ScatterFaces << <gridSize, blockSize >> > (size, thrust::raw_pointer_cast(&mortonsToSort[0]), thrust::raw_pointer_cast(&resultsIndices[0]), thrust::raw_pointer_cast(&resultsFaces[0]),faces);
			CudaCheck(cudaPeekAtLastError());
			CudaCheck(cudaDeviceSynchronize());
		}
	}

	bvhHost.Build(faceAmount, bvhData, faces, vertices,objects);

	//clear the lists
	addList.clear();
	cameraList.clear();
}

void Scene::Compile() {

	if (addList.size() > 0 || cameraList.size() > 0) {

		uint faceAmountPrevious = faceAmount;
		uint vertexAmountPrevious = vertexAmount;
		uint tetAmountPrevious = tetAmount;
		uint materialAmountPrevious = materialAmount;
		uint objectAmountPrevious = objectAmount;

		for (int i = 0; i < addList.size(); ++i) {

			tetAmount += addList[i].second->tetAmount;
			faceAmount += addList[i].second->faceAmount;
			vertexAmount += addList[i].second->verticeAmount;
			materialAmount += addList[i].second->materialAmount;
			++objectAmount;

		}

		for (int i = 0; i < cameraList.size(); ++i) {
			++faceAmount;
			++vertexAmount;
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
			sceneBox.max = glm::max(sceneBox.max, addList[i].second->box.max);
			sceneBox.min = glm::min(sceneBox.min, addList[i].second->box.min);

			//create the modified host data to upload
			std::vector<Vertex> tempVertices(addList[i].second->verticeAmount);
			std::vector<Face> tempFaces(addList[i].second->faceAmount);
			std::vector<Tet> tempTets(addList[i].second->tetAmount);
			std::vector<Material> tempMaterials(addList[i].second->materialAmount);

			//create the minified object from the input object
			MiniObject tempObject(*addList[i].second);
			CudaCheck(cudaMalloc((void**)&tempObject.transforms, sizeof(SceneNode)*addList[i].first.size()));
			CudaCheck(cudaMemcpy(tempObject.transforms, addList[i].first.data(), sizeof(SceneNode)*addList[i].first.size(), cudaMemcpyHostToDevice));
			tempObject.tSize = addList[i].first.size();

			uint maxIter = glm::max(addList[i].second->materialAmount, glm::max(addList[i].second->verticeAmount, glm::max(addList[i].second->faceAmount, addList[i].second->tetAmount)));

			for (uint t = 0; t < maxIter; ++t) {
				if (t < addList[i].second->verticeAmount) {
					tempVertices[t] = addList[i].second->vertices[t];
					tempVertices[t].position = glm::vec3(tempVertices[t].position);
					tempVertices[t].object = objectOffset;
				}
				if (t < addList[i].second->faceAmount) {
					tempFaces[t] = addList[i].second->faces[t];

					tempFaces[t].indices.x += vertexOffset;
					tempFaces[t].indices.y += vertexOffset;
					tempFaces[t].indices.z += vertexOffset;

					tempFaces[t].material += materialOffset;
				}
				if (t < addList[i].second->tetAmount) {
					tempTets[t] = addList[i].second->tets[t];
					tempTets[t].material += materialOffset;
					tempTets[t].object = objectOffset;
				}
				if (t < addList[i].second->materialAmount) {
					tempMaterials[t] = addList[i].second->materials[t];
				}
			}


			//upload the data
			CudaCheck(cudaMemcpy(vertices + vertexOffset, tempVertices.data(), tempVertices.size() * sizeof(Vertex), cudaMemcpyHostToDevice));
			CudaCheck(cudaMemcpy(faces + faceOffset, tempFaces.data(), tempFaces.size() * sizeof(Face), cudaMemcpyHostToDevice));
			CudaCheck(cudaMemcpy(tets + tetOffset, tempTets.data(), tempTets.size() * sizeof(Tet), cudaMemcpyHostToDevice));
			CudaCheck(cudaMemcpy(materials + materialOffset, tempMaterials.data(), tempMaterials.size() * sizeof(Material), cudaMemcpyHostToDevice));
			CudaCheck(cudaMemcpy(objects + objectOffset, &tempObject, sizeof(MiniObject), cudaMemcpyHostToDevice));


			//update the offsets
			tetOffset += addList[i].second->tetAmount;
			faceOffset += addList[i].second->faceAmount;
			vertexOffset += addList[i].second->verticeAmount;
			materialOffset += addList[i].second->materialAmount;
			++objectOffset;

		}

		for (int i = 0; i < cameraList.size(); ++i) {

			MiniObject obj;
			CudaCheck(cudaMalloc((void**)&obj.transforms, sizeof(SceneNode)*cameraList[i].first.size()));
			CudaCheck(cudaMemcpy(obj.transforms, cameraList[i].first.data(), sizeof(SceneNode)*cameraList[i].first.size(), cudaMemcpyHostToDevice));
			obj.tSize = cameraList[i].first.size();

			Vertex vertex;

			vertex.position = glm::vec3(cameraList[i].second->Position());
			vertex.object = objectOffset;

			Face face;
			face.material = 0;
			face.indices.x = vertexOffset;
			face.indices.y = vertexOffset;
			face.indices.z = vertexOffset;

			cameraList[i].second->currentVert = vertex;
			cameraList[i].second->devicePos = vertices + vertexOffset;

			CudaCheck(cudaMemcpy(vertices + vertexOffset, &vertex, sizeof(Vertex), cudaMemcpyHostToDevice));
			CudaCheck(cudaMemcpy(faces + faceOffset, &face, sizeof(Face), cudaMemcpyHostToDevice));
			CudaCheck(cudaMemcpy(objects + objectOffset, &obj, sizeof(MiniObject), cudaMemcpyHostToDevice));

			++vertexOffset;
			++faceOffset;
			++objectOffset;

		}
	}
}

//object pointer is host
void Scene::AddObject(std::vector<SceneNode> matrix, Object* obj) {
	addList.push_back(std::make_pair(matrix, obj));
}

void Scene::AddCamera(std::vector<SceneNode> matrix, Camera* camera) {
	cameraList.push_back(std::make_pair(matrix, camera));
}

void Scene::RemoveObject(Object* obj) {
	removeList.push_back(obj);
	S_LOG_WARNING("Removal of objects from a scene not yet implemented");
}
