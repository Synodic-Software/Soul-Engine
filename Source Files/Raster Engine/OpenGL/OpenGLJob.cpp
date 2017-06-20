//---------------------------------------------------------------------------------------------------
//@file	N:\Documents\Soul Engine\Source Files\Raster Engine\OpenGL\OpenGLJob.cpp.
//Implements the open gl job class.

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

//Default constructor.
OpenGLJob::OpenGLJob()
	: RasterJob() {

	Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [this]() {
		RasterBackend::MakeContextCurrent();

		//create the program object
		object = glCreateProgram();

		if (object == 0) {
			S_LOG_FATAL("glCreateProgram failed");
			assert(object != 0);
		}

	});
	Scheduler::Block();
}

//Destructor.
OpenGLJob::~OpenGLJob() {

}

//---------------------------------------------------------------------------------------------------
//should be called in the context of the main thread.
//@param	attribName	Name of the attribute.
//@return	The attribute.

GLint OpenGLJob::GetAttribute(const GLchar* attribName) {

	GLint attrib = glGetAttribLocation(object, attribName);
	if (attrib == -1) {
		S_LOG_FATAL("Program attribute not found: ", attribName);
	}

	return attrib;
}

//---------------------------------------------------------------------------------------------------
//Attach shaders.
//@param	shadersIn	The shaders in.

void OpenGLJob::AttachShaders(const std::vector<Shader*>& shadersIn) {
	shaders = shadersIn;
	OpenGLJob* job = this;
	Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [&job]() {

		RasterBackend::MakeContextCurrent();

		//attach all the shaders
		for (unsigned i = 0; i < job->shaders.size(); ++i) {
			OpenGLShader* temp = static_cast<OpenGLShader*>(job->shaders[i]);
			glAttachShader(job->object, temp->Object());
		}

		//link the shaders together
		glLinkProgram(job->object);

		//detach all the shaders
		for (unsigned i = 0; i < job->shaders.size(); ++i) {
			OpenGLShader* temp = static_cast<OpenGLShader*>(job->shaders[i]);
			glDetachShader(job->object, temp->Object());
		}

		//throw exception if linking failed
		GLint status;
		glGetProgramiv(job->object, GL_LINK_STATUS, &status);
		if (status == GL_FALSE) {

			std::string msg("Program linking failure in: ");
			GLint infoLogLength;
			glGetProgramiv(job->object, GL_INFO_LOG_LENGTH, &infoLogLength);
			char* strInfoLog = new char[infoLogLength + 1];
			glGetProgramInfoLog(job->object, infoLogLength, nullptr, strInfoLog);
			msg += strInfoLog;
			delete[] strInfoLog;

			glDeleteProgram(job->object); job->object = 0;
			S_LOG_FATAL(msg);

		}
	});

	Scheduler::Block();
}

//---------------------------------------------------------------------------------------------------
//Registers the uniform described by uniformName.
//@param	uniformName	Name of the uniform.

void OpenGLJob::RegisterUniform(const std::string uniformName) {

	GLint uniform;

	Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [this, &uniform,&uniformName]() {
		RasterBackend::MakeContextCurrent();

		uniform = glGetUniformLocation(object, uniformName.c_str());
		if (uniform == -1) {
			S_LOG_FATAL("Program uniform not found: ", uniformName);
		}

	});
	Scheduler::Block();

	(*this)[uniformName] = uniform;
}

//---------------------------------------------------------------------------------------------------
//Uploads a geometry.
//@param [in,out]	vertices   	If non-null, the vertices.
//@param 		 	verticeSize	Size of the vertice.
//@param [in,out]	indices	   	If non-null, the indices.
//@param 		 	indiceSize 	Size of the indice.

void OpenGLJob::UploadGeometry(float* vertices, uint verticeSize, uint* indices, uint indiceSize) {

	Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [this, &vertices, &indices, verticeSize, indiceSize]() {
		RasterBackend::MakeContextCurrent();

		const size_t VertexSize = sizeof(GLfloat) * 4;

		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, verticeSize, vertices, GL_STATIC_DRAW);

		GLint attrib = GetAttribute("vert_VS_in");

		glEnableVertexAttribArray(attrib);
		glVertexAttribPointer(attrib, 4, GL_FLOAT, GL_FALSE, VertexSize, nullptr);

		glGenBuffers(1, &ibo);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indiceSize, indices, GL_STATIC_DRAW);

		glBindVertexArray(0);

		drawSize = indiceSize / sizeof(uint);

	});
	Scheduler::Block();
}

//---------------------------------------------------------------------------------------------------
//Sets an uniform.
//@param	uniformName	Name of the uniform.
//@param	type	   	The type.

void OpenGLJob::SetUniform(const std::string uniformName, RasterVariant type) {

	Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [this, &uniformName, type]() {
		RasterBackend::MakeContextCurrent();

		glUseProgram(object);
		glBindVertexArray(vao);

		int index = type.which();
		if (index == 1) {

		}
		else if (index == 2) {

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
			glUniform2uiv(shaderUniforms[uniformName], 1, (const GLuint*)&boost::get<glm::uvec2>(type));
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
	});
	Scheduler::Block();

}

//Draws this object.
void OpenGLJob::Draw() {

	OpenGLJob* job = this;

	Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [&job]() {
		RasterBackend::MakeContextCurrent();

		glUseProgram(job->object);
		glBindVertexArray(job->vao);
		glDrawElements(GL_TRIANGLES, job->drawSize, GL_UNSIGNED_INT, (GLvoid*)0);
		glBindVertexArray(0);
		glUseProgram(0);

	});
	Scheduler::Block();
}
