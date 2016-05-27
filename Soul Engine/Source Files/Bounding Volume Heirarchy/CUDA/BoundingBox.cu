#include "BoundingBox.cuh"

BoundingBox::BoundingBox()
{
}

BoundingBox::BoundingBox(glm::vec3 minN, glm::vec3 maxN)
{
	min = minN;
	max = maxN;
}

BoundingBox::~BoundingBox()
{
}
