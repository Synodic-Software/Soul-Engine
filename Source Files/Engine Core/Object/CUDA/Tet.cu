#include "Tet.cuh"


Tet::Tet()
{
}
Tet::Tet(glm::uvec4 ind, uint matID){
	indices = ind;
	material = matID;
}

Tet::~Tet()
{
}
