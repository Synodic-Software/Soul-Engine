#include "Face.cuh"


Face::Face()
{
}
Face::Face(glm::uvec3 ind, Material* matID){
	indices = ind;
	materialID = matID;
}

void Face::SetData(glm::uvec3 ind, Material* matID){
	indices = ind;
	materialID = matID;
}

Face::~Face()
{
}
