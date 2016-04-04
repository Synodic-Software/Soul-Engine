#include "Face.cuh"


Face::Face()
{
}
Face::Face(glm::uvec3 ind, Material* matID){
	indices = ind;
	materialPointer = matID;
}

void Face::SetData(glm::uvec3 ind, Material* matID){
	indices = ind;
	materialPointer = matID;
}

Face::~Face()
{
}
