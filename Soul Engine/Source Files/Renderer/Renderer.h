#pragma once

#include "Engine Core/Camera/Camera.h"
#include "Engine Core/BasicDependencies.h"
#include "Bounding Volume Heirarchy/BVH.h"

class Renderer{
public:
	void Render(glm::uvec2, BVH*, Camera&, uint);
	Renderer();
private:

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


	shading::ShaderSupport* CUDAtoScreen;


	GLuint displayTexture;
	CUarray cudaDisplay;
	CUgraphicsResource graphicsResource;
};
