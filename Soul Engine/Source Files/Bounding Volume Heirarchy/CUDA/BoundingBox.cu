#include "BoundingBox.cuh"

BoundingBox::BoundingBox()
{
}

BoundingBox::BoundingBox(glm::vec3 originN, glm::vec3 extentN)
{
	origin = originN;
	extent = extentN;
}

BoundingBox::~BoundingBox()
{
}
