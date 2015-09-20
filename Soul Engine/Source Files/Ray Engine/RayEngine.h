#pragma once

#include "Engine Core/BasicDependencies.h"
#include "Utility/OpenGL/ShaderSupport.h"
#include "Utility/OpenGL/Shader.h"
#include "Engine Core/Material/Texture/Texture.h"
#include "Renderer/Renderer.h"
#include "RayJob.h"

class RayEngine {
public:

	RayEngine(glm::uvec2);
	void Render(glm::uvec2, BVH*, Camera&, double);
	void AddJob(); 
	void AddRecurringJob();
private:




	Renderer* mainTracer;

	std::vector<RayJobs> jobs;

};
