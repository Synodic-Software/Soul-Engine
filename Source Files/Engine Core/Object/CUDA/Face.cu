#include "Face.cuh"


Face::Face()
{
}
Face::Face(glm::uvec3 ind, uint matID){
	indices = ind;
	material = matID;
}

Face::~Face()
{
}
