#pragma once

#include <string>

#include "OpenGLBackend.h"
#include "Raster Engine\RasterJob.h"

class OpenGLJob : public RasterJob {
public:
	OpenGLJob();
	~OpenGLJob();

	void AttachShaders(const std::vector<Shader*>&);

private:
	GLuint vao;
	GLuint vbo;
	GLuint ibo;
};