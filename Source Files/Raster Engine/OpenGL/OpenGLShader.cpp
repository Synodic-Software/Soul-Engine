#include "OpenGLShader.h"

#include <stdexcept>
#include <fstream>
#include <iostream>
#include <string>
#include <cassert>
#include <sstream>

#include "Multithreading\Scheduler.h"
#include "Utility\Logger.h"

#include "Raster Engine\RasterBackend.h"

/*
 *    Constructor.
 *
 *    @param	filePath  	Full pathname of the file.
 *    @param	shaderType	Type of the shader.
 */

OpenGLShader::OpenGLShader(std::string filePath, shader_t shaderType)
	: Shader(filePath, shaderType), object(0) {
	Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [&object = object, &name = name, &codeStr = codeStr, shaderType]() {

		RasterBackend::MakeContextCurrent();

		//create the shader object

		GLenum shaderT;

		switch (shaderType) {
		case VERTEX_SHADER:
			shaderT = GL_VERTEX_SHADER;
			break;
		case FRAGMENT_SHADER:
			shaderT = GL_FRAGMENT_SHADER;
			break;

		}


		object = glCreateShader(shaderT);
		if (object == 0) {
			S_LOG_FATAL("glCreateShader failed");
		}

		//set the source code
		const GLchar* code = (const GLchar*)codeStr.c_str();
		glShaderSource(object, 1, &code, 0);

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
			glGetShaderInfoLog(object, infoLogLength, nullptr, strInfoLog);
			msg += strInfoLog;
			delete[] strInfoLog;

			glDeleteShader(object); object = 0;

			S_LOG_FATAL(msg);
		}


	});

	Scheduler::Block();
}

/* Destructor. */
/* Destructor. */
OpenGLShader::~OpenGLShader() {

	Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [&object = object]() {

		RasterBackend::MakeContextCurrent();
		glDeleteShader(object); object = 0;

	});

	Scheduler::Block();

}

/*
 *    Copy constructor.
 *
 *    @param	other	The other.
 */

OpenGLShader::OpenGLShader(const OpenGLShader& other) :
	Shader(other.name, other.type),
	object(other.object)
{
	referenceCount = other.referenceCount;
	name = other.name;
	Retain();
}

/*
 *    Gets the object.
 *
 *    @return	A GLuint.
 */

GLuint OpenGLShader::Object() const {
	return object;
}

/*
 *    Assignment operator.
 *
 *    @param	other	The other.
 *
 *    @return	A shallow copy of this object.
 */

OpenGLShader& OpenGLShader::operator = (const OpenGLShader& other) {
	Release();
	object = other.object;
	referenceCount = other.referenceCount;
	Retain();
	return *this;
}