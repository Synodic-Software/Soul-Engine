#include "Tet.cuh"


Tet::Tet()
{
}
Tet::Tet(glm::uvec4 ind, Material* matID){
	indices = ind;
	materialPointer = matID;
}

Tet::~Tet()
{
}
