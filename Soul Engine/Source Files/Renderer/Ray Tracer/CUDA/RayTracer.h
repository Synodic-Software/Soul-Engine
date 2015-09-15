#pragma once

#include "Engine Core/BasicDependencies.h"
#include "Utility/OpenGL/ShaderSupport.h"
#include "Utility/OpenGL/Shader.h"
#include "Engine Core/Material/Texture/Texture.h"
#include "Bounding Volume Heirarchy/BVH.h"
#include "Renderer/Renderer.h"

class RayTracer: public Renderer{
public:
	RayTracer(glm::uvec2);
	GLuint Draw(glm::uvec2, BVH*, Camera&,GLuint);

private:
};

