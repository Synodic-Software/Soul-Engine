#include "Face.h"


Face::Face()
{
}
Face::Face(glm::uvec3 ind, uint matID){
	indices = ind;
	materialID = matID;
}

Face::~Face()
{
}
