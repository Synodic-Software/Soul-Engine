#pragma once

#include <string>

#include "OpenGLBackend.h"
#include "Raster Engine\Shader.h"

class OpenGLShader : public Shader {
public:
	OpenGLShader(std::string, shader_t);
	OpenGLShader(const OpenGLShader& other);
	virtual ~OpenGLShader();

	GLuint Object() const;
	OpenGLShader& operator =(const OpenGLShader& other);

private:
	GLuint object;
};