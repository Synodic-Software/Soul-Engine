#pragma once

#include "Utility\Includes\GLMIncludes.h"
#include "Metrics.h"

class SceneNode
{

public:

	SceneNode(int, glm::mat4);

	int scale; 
	uint64 mortonMin;
	uint64 mortonMax;

private:
	
};


