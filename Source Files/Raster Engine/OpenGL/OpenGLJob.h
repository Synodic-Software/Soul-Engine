#pragma once

#include <string>

#include "OpenGLBackend.h"
#include "Raster Engine\RasterJob.h"

class OpenGLJob : public RasterJob {
public:
	OpenGLJob(const std::vector<Shader>&);
	~OpenGLJob();

private:
	GLuint vao;
	GLuint vbo;
	GLuint ibo;
};