#pragma once

#include "Engine Core\BasicDependencies.h"
#include "BoundingBox.cuh"
class Node : public Managed
{
public:
	Node();
	~Node();

private:
	float systemMin = 0.0f;
	float systemMax = 8192.0f;
	BoundingBox* box;

};