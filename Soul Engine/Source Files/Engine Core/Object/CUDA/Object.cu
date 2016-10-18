#include "Object.cuh"
#include "Utility\CUDA\CUDAHelper.cuh"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

Object::Object(){

	verticeAmount = 0;
	faceAmount = 0;
	materialSize = 0;
	localSceneIndex = 0;
	ready = false;

	xyzPosition = glm::vec3(0);

	vertices = NULL;
	faces = NULL;
	materialP = NULL;
}
Object::Object(glm::vec3 pos, std::string name, Material* mat){

	verticeAmount = 0;
	faceAmount = 0;
	materialSize = 1;
	localSceneIndex = 0;
	ready = false;

	xyzPosition = glm::vec3(0);

	vertices = NULL;
	faces = NULL;
	CudaCheck(cudaMallocManaged((void**)&materialP, materialSize*sizeof(Material*)));
	materialP[0] = mat;

	xyzPosition = pos;
	ExtractFromFile(name.c_str());
}

void Object::AddVertices(Vertex* vertices, uint vSize){

}
void Object::AddFaces(Face* vertices, uint fSize){

}
void Object::ExtractFromFile(const char* name){



	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err;

	if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err, name)) {
		throw std::runtime_error(err);
	}

	//
	uint overallSize = 0;
	uint faceOverallSize = 0;
	for (uint i = 0; i < shapes.size(); i++){
		overallSize += attrib.vertices.size() / 3;
		faceOverallSize += shapes[i].mesh.indices.size() / 3;
	}


	verticeAmount = overallSize;
	faceAmount = faceOverallSize;

	glm::vec3 max = glm::vec3(attrib.vertices[0], attrib.vertices[1], attrib.vertices[2]);
	glm::vec3 min = max;

	cudaDeviceSynchronize();

	CudaCheck(cudaMallocManaged((void**)&vertices,
		verticeAmount*sizeof(Vertex)));

	CudaCheck(cudaMallocManaged((void**)&faces,
		faceAmount*sizeof(Face)));

	cudaDeviceSynchronize();




	//

	std::unordered_map<Vertex, int> uniqueVertices = {};

	int vertexFillSize = 0;
	int indexFillSize = 0;

	for (const auto& shape : shapes) {
		for (const auto& index : shape.mesh.indices) {
			Vertex vertex = {};

			vertex.position = {
				attrib.vertices[3 * index.vertex_index + 0],
				attrib.vertices[3 * index.vertex_index + 1],
				attrib.vertices[3 * index.vertex_index + 2]
			};

			vertex.textureCoord = {
				attrib.texcoords[2 * index.texcoord_index + 0],
				1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
			};

			vertex.normal = {
				attrib.normals[3 * index.normal_index + 0],
				attrib.normals[3 * index.normal_index + 1],
				attrib.normals[3 * index.normal_index + 2]
			};


			if (uniqueVertices.count(vertex) == 0) {

				uniqueVertices[vertex] = vertexFillSize;
				vertices[vertexFillSize]=vertex;

				vertices[vertexFillSize].position += xyzPosition;
				max = glm::max(vertices[vertexFillSize].position, max);

				min = glm::min(vertices[vertexFillSize].position, min);
				vertexFillSize++;
			}

			if (indexFillSize%3==0){
				faces[indexFillSize].indices.x = uniqueVertices[vertex];

			}
			else if (indexFillSize%3==1){
				faces[indexFillSize].indices.y = uniqueVertices[vertex];

			}
			else{
				faces[indexFillSize].indices.z = uniqueVertices[vertex];
				faces[indexFillSize].materialPointer= materialP[0];
			}
			indexFillSize++;
		}
	}

	box.max = max;
	box.min = min;
	cudaDeviceSynchronize();

}