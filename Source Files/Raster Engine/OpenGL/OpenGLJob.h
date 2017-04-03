#pragma once

#include <string>

#include "OpenGLBackend.h"
#include "Raster Engine\RasterJob.h"

class OpenGLJob : public RasterJob {
public:
	OpenGLJob();
	~OpenGLJob();

	void AttachShaders(const std::vector<Shader*>&);
	void RegisterUniform(const std::string);
	void UploadGeometry(float*, uint, uint*, uint);
	void SetUniform(const std::string, RasterVariant);

	void Draw();


private:
	GLint GetAttribute(const GLchar*);

	GLuint vao;
	GLuint vbo;
	GLuint ibo;

	GLuint object;

	uint drawSize;
};