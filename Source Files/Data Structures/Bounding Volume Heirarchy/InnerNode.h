#pragma once

#include "Data Structures/Geometric Primatives/BoundingBox.h"
#include "Metrics.h"

class InnerNode
{

public:

	BoundingBox box;

	uint childLeft;
	uint childRight;

	uint flags;
	uint atomic;

	uint rangeRight;
	uint rangeLeft;
	
};