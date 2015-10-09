#pragma once

#include "Engine Core/Camera/CUDA/Camera.cuh"
#include "Engine Core/BasicDependencies.h"
#include "Bounding Volume Heirarchy/BVH.h"
#include "Utility/OpenGL/ShaderSupport.h"
#include "Utility/OpenGL/Shader.h"
#include "Engine Core/Material/Texture/Texture.h"
#include "Ray Engine/RayEngine.h"

class Renderer{
public:

	Renderer(Camera&, glm::uvec2);


	void RenderRequestChange(glm::uvec2, Camera&, double);
	void Render();


private:
	RayJob* RenderJob;
	double prevTime;
	GLuint calcPass;
	float changeCutoff;
	GLuint samplesMax;
	GLuint samplesMin;
	GLuint samples;
	glm::uvec2 originalScreen;
	glm::uvec2 modifiedScreen;

	GLuint vao;
	GLuint vbo;
	GLuint ibo;

	GLuint Indices[6];

	float Vertices[6 * 4];
	GLint texUniform;
	GLint screenUniform;
	GLint cameraUniform;
	GLint modelUniform;
	GLint screenModUniform;


	shading::ShaderSupport* CUDAtoScreen;

	GLuint renderBuffer;
	struct cudaGraphicsResource *cudaBuffer;
	float4 *bufferData;
};
