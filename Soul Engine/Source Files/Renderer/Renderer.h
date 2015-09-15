#pragma once

#include "Engine Core/Camera/Camera.h"
#include "Engine Core/BasicDependencies.h"
#include "Bounding Volume Heirarchy/BVH.h"

class Renderer{
public:
	void Draw(glm::uvec2, BVH*, Camera, uint);

private:

};
