#pragma once

#include "Utility\Includes\GLMIncludes.h"
#include "Metrics.h"

class SceneNode
{

public:

	SceneNode(glm::mat4);

	int bitsUsed;
	uint64 morton;

private:
	
};


