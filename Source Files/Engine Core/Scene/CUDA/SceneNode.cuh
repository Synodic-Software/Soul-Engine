#pragma once

#include "Utility\Includes\GLMIncludes.h"
#include "Engine Core\Object\MiniObject.h"

class SceneNode
{

public:

	SceneNode(int, glm::mat4);

	int scale;
	glm::mat4 transform; //bounded by unit box

private:
	
};


