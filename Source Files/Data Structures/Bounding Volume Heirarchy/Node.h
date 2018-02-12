#pragma once

#include "Data Structures/Geometric Primatives/BoundingBox.h"
#include "Metrics.h"

class Node
{

public:

	BoundingBox box;

	uint childLeft;
	uint rangeLeft;

	uint childRight;
	uint rangeRight;

	uint atomic;

};