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

OpenGLJob::OpenGLJob()
	: RasterJob() {

	Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [this]() {
		RasterBackend::RasterFunction([this]() {
			//create the program object
			object = glCreateProgram();

			if (object == 0) {
				S_LOG_FATAL("glCreateProgram failed");
			}
		});

	});
	Scheduler::Block();
}

OpenGLJob::~OpenGLJob() {

}

void OpenGLJob::AttachShaders(const std::vector<Shader*>& shaders) {

	Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [this, &shaders]() {

		RasterBackend::RasterFunction([this, &shaders]() {

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
	});

	Scheduler::Block();
}
