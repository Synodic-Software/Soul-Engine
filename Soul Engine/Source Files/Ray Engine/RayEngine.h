#pragma once

#include "Engine Core/BasicDependencies.h"
#include "Utility/OpenGL/ShaderSupport.h"
#include "Utility/OpenGL/Shader.h"
#include "Engine Core/Material/Texture/Texture.h"
#include "Renderer/Renderer.h"

class RayEngine{
public:
	typedef struct Job{
		int type;
	}Job;

	RayEngine(glm::uvec2);
	void Render(glm::uvec2, BVH*, Camera&, double);

private:

	GLuint displayTexture;
	CUarray cudaDisplay;
	CUgraphicsResource graphicsResource;





	double prevTime;
	GLuint calcPass;
	float changeCutoff;
	GLuint samplesMax;
	GLuint samplesMin;
	GLuint samples;


	GLuint vao;
	GLuint vbo;
	GLuint ibo;

	GLuint Indices[6];

	float Vertices[6 * 4];
	GLint texUniform;
	GLint screenUniform;
	GLint cameraUniform;
	GLint modelUniform;

	Renderer* mainTracer;

	std::vector<Job> jobs;
	shading::ShaderSupport* CUDAtoScreen;
	//shading::ShaderSupport* generateRays;
	//shading::ShaderSupport* sendRays;

};
