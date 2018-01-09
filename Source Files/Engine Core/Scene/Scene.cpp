#include "Scene.h"

#define RAY_BIAS_DISTANCE 0.0002f 
#define BVH_STACK_SIZE 64
#define DYNAMIC_FETCH_THRESHOLD 20          // If fewer than this active, fetch new rays

#include "Utility\Logger.h"

#include "Algorithms\Morton Code\MortonCode.h"

#include "GPGPU/GPUManager.h"
#include "Algorithms/Data Algorithms/GPU Sort/Sort.h"

Scene::Scene()
{
	bvhData.Move(GPUManager::GetBestGPU());
	sky.Move(GPUManager::GetBestGPU());
	faces.Move(GPUManager::GetBestGPU());
	vertices.Move(GPUManager::GetBestGPU());
	tets.Move(GPUManager::GetBestGPU());
	materials.Move(GPUManager::GetBestGPU());
	objects.Move(GPUManager::GetBestGPU());
	bvh.Move(GPUManager::GetBestGPU());
	mortonCodes.Move(GPUManager::GetBestGPU());

	bvh.resize(1);
	bvhData.resize(1);
	sky.push_back({ "Starmap.png" });

	bvh.TransferToDevice();
	bvhData.TransferToDevice();
	sky.TransferToDevice();
}

__host__ void Scene::Build(float deltaTime) {

	Compile();

	auto size = faces.HostSize();
	if (size > 0) {

		GPUDevice device = GPUManager::GetBestGPU();

		const auto blockSize = 64;
		const GPUExecutePolicy normalPolicy(glm::vec3((size + blockSize - 1) / blockSize, 1, 1), glm::vec3(blockSize, 1, 1), 0, 0);

		auto mortP = mortonCodes.DeviceData();
		auto faceP = faces.DeviceData();
		auto vertP = vertices.DeviceData();

		device.Launch(normalPolicy, MortonCode::ComputeGPUFace64, size, mortP, faceP, vertP);


		Sort::Sort(mortonCodes, faces);

	}

	bvh[0].Build(size, bvhData, mortonCodes, faces, vertices);

}

void Scene::Compile() {

	//upload the data
	vertices.TransferToDevice();
	faces.TransferToDevice();
	tets.TransferToDevice();
	materials.TransferToDevice();
	objects.TransferToDevice();

}

//object pointer is host
void Scene::AddObject(Object& obj) {

	auto faceAmount = faces.HostSize();
	auto vertexAmount = vertices.HostSize();
	auto tetAmount = tets.HostSize();
	auto materialAmount = materials.HostSize();
	auto objectAmount = objects.HostSize();

	const auto faceOffset = faceAmount;
	const auto vertexOffset = vertexAmount;
	const auto tetOffset = tetAmount;
	const auto materialOffset = materialAmount;
	const auto objectOffset = objectAmount;

	tetAmount += obj.tetAmount;
	faceAmount += obj.faceAmount;
	vertexAmount += obj.verticeAmount;
	materialAmount += obj.materialAmount;
	++objectAmount;

	//resizing	
	vertices.resize(vertexAmount);
	faces.resize(faceAmount);
	mortonCodes.resize(faceAmount);
	tets.resize(tetAmount);
	materials.resize(materialAmount);
	objects.resize(objectAmount);


	//update the scene's bounding volume
	sceneBox.max = glm::max(sceneBox.max, obj.box.max);
	sceneBox.min = glm::min(sceneBox.min, obj.box.min);

	//create the minified object from the input object
	objects[objectOffset] = MiniObject(obj);

	const auto maxIter =
		glm::max(obj.materialAmount,
			glm::max(obj.verticeAmount,
				glm::max(obj.faceAmount,
					obj.tetAmount
				)));

	for (uint t = 0; t < maxIter; ++t) {
		if (t < obj.verticeAmount) {
			vertices[t + vertexOffset] = obj.vertices[t];
			vertices[t + vertexOffset].position = glm::vec3(vertices[t].position);
			vertices[t + vertexOffset].object = objectOffset;
		}
		if (t < obj.faceAmount) {
			faces[t + faceOffset] = obj.faces[t];

			faces[t + faceOffset].indices.x += vertexOffset;
			faces[t + faceOffset].indices.y += vertexOffset;
			faces[t + faceOffset].indices.z += vertexOffset;

			faces[t + faceOffset].material += materialOffset;
		}
		if (t < obj.tetAmount) {
			tets[t + tetOffset] = obj.tets[t];
			tets[t + tetOffset].material += materialOffset;
			tets[t + tetOffset].object = objectOffset;
		}
		if (t < obj.materialAmount) {
			materials[t + objectOffset] = obj.materials[t];
		}
	}



}

void Scene::RemoveObject(Object& obj) {

	S_LOG_WARNING("Removal of objects from a scene not yet implemented");
}
