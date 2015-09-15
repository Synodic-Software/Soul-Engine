#include "ObjectProperties.h"

ObjectProperties::ObjectProperties(){
	vertices = new Vertex();
	//instances = new Instance();
	translate = new float3(make_float3(0.0f, 0.0f, 0.0f));
	rotate = new float3(make_float3(0.0f, 0.0f,0.0f));
	scale = new float3(make_float3(0.0f, 0.0f, 0.0f));
}
void ObjectProperties::AddVertices(std::vector<Vertex>& verticesAdd){
	vertices = new Vertex[verticesAdd.size()];
	for (int i = 0; i < verticesAdd.size(); i++){
		vertices[i] = verticesAdd[i];
	}
}
void ObjectProperties::AddFaces(std::vector<Face>& facesAdd){
	faces = new Face[facesAdd.size()];
	for (int i = 0; i < facesAdd.size(); i++){
		faces[i] = facesAdd[i];
	}
}