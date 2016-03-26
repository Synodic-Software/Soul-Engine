#include "Object.cuh"


Object::Object(){

	verticeAmount=0;
	faceAmount=0;
	materialSize = 0;
	localSceneIndex = 0;
	ready = false;

	xyzPosition = glm::vec3(0);

	vertices=NULL;
	faces = NULL;
	materials = NULL;
}

void Object::AddVertices(Vertex* vertices, uint vSize){
	
}
void Object::AddFaces(Face* vertices, uint fSize){

}
void Object::ExtractFromFile(const char* name){
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err = tinyobj::LoadObj(shapes, materials, name, NULL);

	if (!err.empty()) {
		std::cerr << err << std::endl;
	}
	
	uint overallSize = 0;
	uint faceOverallSize = 0;
	for (uint i = 0; i < shapes.size(); i++){
		overallSize += shapes[i].mesh.positions.size();
		faceOverallSize += shapes[i].mesh.indices.size();
	}


	verticeAmount = overallSize;
	faceAmount = faceOverallSize;

	glm::vec3 max = glm::vec3(shapes[0].mesh.positions[0], shapes[0].mesh.positions[1], shapes[0].mesh.positions[2]);
	glm::vec3 min = glm::vec3(shapes[0].mesh.positions[0], shapes[0].mesh.positions[1], shapes[0].mesh.positions[2]);

	cudaDeviceSynchronize();

	cudaMallocManaged(&vertices,
		verticeAmount*sizeof(Vertex));

	cudaMallocManaged(&faces,
		faceAmount*sizeof(Face));

	cudaDeviceSynchronize();

	uint overallOffset = 0;
	uint faceOffset = 0;

	for (uint i = 0; i < shapes.size(); i++){
		for (uint v = 0; v < verticeAmount / 3; v++){
			vertices[overallOffset + v].SetData(
				glm::vec3(shapes[i].mesh.positions[3 * v + 0], shapes[i].mesh.positions[3 * v + 1], shapes[i].mesh.positions[3 * v + 2])*METER,
				glm::vec2(shapes[i].mesh.texcoords[2 * v + 0], shapes[i].mesh.texcoords[2 * v + 1]),
				glm::vec3(shapes[i].mesh.normals[3 * v + 0], shapes[i].mesh.normals[3 * v + 1], shapes[i].mesh.normals[3 * v + 2]));

			vertices[overallOffset + v].position += xyzPosition;
			max = glm::max(vertices[overallOffset + v].position, max);
			min = glm::min(vertices[overallOffset + v].position, min);

		}
		overallOffset += shapes[i].mesh.positions.size();

		for (uint f = 0; f < faceAmount / 3; f++){
			faces[faceOffset + f].SetData(
				glm::uvec3(shapes[i].mesh.indices[3 * f + 0], shapes[i].mesh.indices[3 * f + 1], shapes[i].mesh.indices[3 * f + 2]),
				NULL,this);
				//shapes[i].mesh.material_ids[f]);
		}
		faceOffset += shapes[i].mesh.indices.size();
	}

	box.origin = ((max - min) / 2.0f) + min;
	box.extent = box.origin - min;
	cudaDeviceSynchronize();
}
//std::string Object::ResourcePath(std::string fileName) {
//		return  fileName;
//}
//

//void Object::GetVertices(GLuint& buffer, GLuint& sizeVert){
//	buffer = storageVertices;
//	sizeVert = verticeAmount;
//}
//void Object::GetIndices(GLuint& buffer, GLuint& sizeIn){
//	buffer = storageIndices;
//	sizeIn = indiceAmount;
//}
//void Object::GetTextureCoords(GLuint& buffer, GLuint& sizeVert){
//	buffer = storageTextureCoords;
//	sizeVert = verticeAmount;
//}
//void Object::GetNormals(GLuint& buffer, GLuint& sizeVert){
//	buffer = storageNormals;
//	sizeVert = verticeAmount;
//}
//void Object::GetMaterials(std::list<Material*>& material){
//	for (std::list<Material>::iterator itr = materials.begin(); itr != materials.end(); itr++){
//		material.push_back(&*itr);
//	}
//}
//void Object::GetPhysics(glm::vec3& pos, glm::vec3& vel,bool& isStat){
//	pos = xyzPosition;
//	vel = velocity;
//	isStat = isStatic;
//}
//void Object::SetPhysics(glm::vec3& pos, glm::vec3& vel, bool& isStat){
//	xyzPosition = pos;
//	velocity = vel;
//	isStatic = isStat;
//}
//Object::Mesh::VertexHandle Object::AddVertex(float x, float y, float z){
//	return mesh.add_vertex(Mesh::Point(x, y, z));
//}
//Object::Mesh::VertexHandle Object::AddVertex(glm::vec3 data){
//	return mesh.add_vertex(Mesh::Point(data.x, data.y, data.z));
//}
//Object::Mesh::VertexHandle Object::AddVertex(glm::vec4 data){
//	return mesh.add_vertex(Mesh::Point(data.x, data.y, data.z));
//}
//void Object::AddFace(Object::Mesh::VertexHandle x, Object::Mesh::VertexHandle y, Object::Mesh::VertexHandle z){
//	std::vector<Mesh::VertexHandle>  face_vhandles;
//	face_vhandles.resize(3);
//	face_vhandles[0] = x;
//	face_vhandles[1] = y;
//	face_vhandles[2] = z;
//	mesh.add_face(face_vhandles);
//}