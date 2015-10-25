#include "Object.cuh"


Object::Object(){

	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err = tinyobj::LoadObj(shapes, materials, filename, basepath);

	if (!err.empty()) {
		std::cerr << err << std::endl;
	}
}

void Object::AddVertices(Vertex* vertices, uint vSize){
	
}
void Object::AddFaces(Face* vertices, uint fSize){

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