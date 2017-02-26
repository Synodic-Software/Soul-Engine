#include "OpenGLShader.h"

#include <stdexcept>
#include <fstream>
#include <iostream>
#include <string>
#include <cassert>
#include <sstream>

OpenGLShader::OpenGLShader(GLFWwindow* window,std::string filePath, shader_t shaderType)
	: Shader(filePath, shaderType), object(0){


	//create the shader object
	object = glCreateShader(shaderType);
	if (object == 0)
		throw std::runtime_error("glCreateShader failed");

	//set the source code
	const char* code = codeStr.c_str();
	glShaderSource(object, 1, (const GLchar**)&code, NULL);

	//compile
	glCompileShader(object);

	//throw exception if compile error occurred
	GLint status;
	glGetShaderiv(object, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE) {
		std::string msg("Compile failure in shader " + name + ":\n");

		GLint infoLogLength;
		glGetShaderiv(object, GL_INFO_LOG_LENGTH, &infoLogLength);
		char* strInfoLog = new char[infoLogLength + 1];
		glGetShaderInfoLog(object, infoLogLength, NULL, strInfoLog);
		msg += strInfoLog;
		delete[] strInfoLog;

		glDeleteShader(object); object = 0;
		std::cerr << msg << std::endl;
		throw std::runtime_error(msg);
	}
}

OpenGLShader::~OpenGLShader() {
	glDeleteShader(object); object = 0;
}
OpenGLShader::OpenGLShader(const OpenGLShader& other) :
	Shader(other.name, other.type),
	object(other.object)
{
	referenceCount = other.referenceCount;
	name = other.name;
	Retain();
}
GLuint OpenGLShader::Object() const {
	return object;
}

OpenGLShader& OpenGLShader::operator = (const OpenGLShader& other) {
	Release();
	object = other.object;
	referenceCount = other.referenceCount;
	Retain();
	return *this;
}