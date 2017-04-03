#include "OpenGLJob.h"

#include <stdexcept>
#include <fstream>
#include <iostream>
#include <string>
#include <cassert>
#include <sstream>

#include "Raster Engine\RasterBackend.h"
#include "Raster Engine\OpenGL\OpenGLShader.h"
#include "Utility\Logger.h"
#include "Multithreading\Scheduler.h"

#include <glm/gtc/type_ptr.hpp>

OpenGLJob::OpenGLJob()
	: RasterJob() {

	Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [&object = object]() {
		RasterBackend::MakeContextCurrent();

		//create the program object
		object = glCreateProgram();

		if (object == 0) {
			S_LOG_FATAL("glCreateProgram failed");
		}

	});
	Scheduler::Block();
}

OpenGLJob::~OpenGLJob() {

}

GLint OpenGLJob::GetAttribute(const GLchar* attribName) {
	if (!attribName)
		throw std::runtime_error("attribName was NULL");

	GLint attrib = glGetAttribLocation(object, attribName);
	if (attrib == -1)
		throw std::runtime_error(std::string("Program attribute not found: ") + attribName);

	return attrib;
}

void OpenGLJob::AttachShaders(const std::vector<Shader*>& shaders) {

	Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [&object = object, &shaders]() {

		RasterBackend::MakeContextCurrent();

		//attach all the shaders
		for (unsigned i = 0; i < shaders.size(); ++i) {
			glAttachShader(object, static_cast<OpenGLShader*>(shaders[i])->Object());
		}

		//link the shaders together
		glLinkProgram(object);

		//detach all the shaders
		for (unsigned i = 0; i < shaders.size(); ++i) {
			glDetachShader(object, static_cast<OpenGLShader*>(shaders[i])->Object());
		}

		//throw exception if linking failed
		GLint status;
		glGetProgramiv(object, GL_LINK_STATUS, &status);
		if (status == GL_FALSE) {

			std::string msg("Program linking failure in: ");
			GLint infoLogLength;
			glGetProgramiv(object, GL_INFO_LOG_LENGTH, &infoLogLength);
			char* strInfoLog = new char[infoLogLength + 1];
			glGetProgramInfoLog(object, infoLogLength, NULL, strInfoLog);
			msg += strInfoLog;
			delete[] strInfoLog;

			glDeleteProgram(object); object = 0;
			S_LOG_FATAL(msg);

		}
	});

	Scheduler::Block();
}

void OpenGLJob::RegisterUniform(const std::string uniformName) {

	GLint uniform = glGetUniformLocation(object, uniformName.c_str());
	if (uniform == -1) {
		S_LOG_FATAL("Program uniform not found: ", uniformName);
	}

	(*this)[uniformName] = uniform;
}

void OpenGLJob::UploadGeometry(float* vertices, uint verticeSize, uint* indices, uint indiceSize) {

	const size_t VertexSize = sizeof(GLfloat) * 4;

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, verticeSize, vertices, GL_STATIC_DRAW);

	GLint attrib = GetAttribute("vert_VS_in");

	glEnableVertexAttribArray(attrib);
	glVertexAttribPointer(attrib, 4, GL_FLOAT, GL_FALSE, VertexSize, NULL);

	glGenBuffers(1, &ibo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indiceSize, indices, GL_STATIC_DRAW);

	glBindVertexArray(0);

	drawSize = indiceSize / sizeof(uint);
}

void OpenGLJob::SetUniform(const std::string uniformName, RasterVariant type) {
	glUseProgram(object);
	glBindVertexArray(vao);

	int index = type.which();
	if (index == 1) {

	}
	else if(index == 2){
			
	}
	else if (index == 3) {

	}
	else if (index == 4) {

	}
	else if (index == 5) {

	}
	else if (index == 6) {

	}
	else if (index == 7) {

	}
	else if (index == 8) {

	}
	else if (index == 9) {

	}
	else if (index == 10) {

	}
	else if (index == 11) {
		glUniform2uiv(shaderUniforms[uniformName],1, (const GLuint*)&boost::get<glm::uvec2>(type));
	}

	/*int,
	float,
			double,
			bool,
			uint,
			glm::mat4,
			glm::vec3,
			glm::uvec3,
			glm::vec4,
			glm::uvec4,
			glm::vec2,
			glm::uvec2*/

	glBindVertexArray(0);
	glUseProgram(0);

}

void OpenGLJob::Draw() {

	glUseProgram(object);
	glBindVertexArray(vao);
	glDrawElements(GL_TRIANGLES, drawSize, GL_UNSIGNED_INT, (GLvoid*)0);
	glBindVertexArray(0);
	glUseProgram(0);

}
