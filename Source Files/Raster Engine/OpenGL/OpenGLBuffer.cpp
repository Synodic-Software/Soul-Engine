#include "OpenGLBuffer.h"

#include "Utility\Logger.h"
#include "OpenGLBackend.h"
#include "Multithreading\Scheduler.h"

#include "Raster Engine\RasterBackend.h"

OpenGLBuffer::OpenGLBuffer(uint sizeInBytes) {

	Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [&buffer = buffer, sizeInBytes]() {

		RasterBackend::MakeContextCurrent();

		glGenBuffers(1, &buffer);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
		glBufferData(GL_SHADER_STORAGE_BUFFER,
			sizeInBytes,
			nullptr, GL_STATIC_DRAW);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	});

	Scheduler::Block();
}

OpenGLBuffer::~OpenGLBuffer() {

}