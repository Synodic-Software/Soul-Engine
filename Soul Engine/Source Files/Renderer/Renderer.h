#pragma once

#include "Engine Core/Camera/Camera.h"
#include "Engine Core/BasicDependencies.h"
#include "Bounding Volume Heirarchy/BVH.h"
#include "Utility/OpenGL/ShaderSupport.h"
#include "Utility/OpenGL/Shader.h"
#include "Engine Core/Material/Texture/Texture.h"
#include "Ray Engine\RayEngine.h"

class Renderer{
public:

	Renderer(Camera&);


	void RenderRequest(glm::uvec2, Camera&, double);
	void Render();

protected:

	RAY_FUNCTION setupFunction;

private:
	RayJob* RenderJob;
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
