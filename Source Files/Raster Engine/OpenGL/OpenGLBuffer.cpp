#include "OpenGLBuffer.h"

#include "Utility\Logger.h"
#include "OpenGLBackend.h"
#include "Multithreading\Scheduler.h"

OpenGLBuffer::OpenGLBuffer(uint sizeInBytes) {

	Scheduler::AddTask(LAUNCH_IMMEDIATE, FIBER_HIGH, true, [&buffer = buffer, sizeInBytes]() {

		RasterBackend::RasterFunction([&buffer = buffer,sizeInBytes]() {

			glGenBuffers(1, &buffer);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
			glBufferData(GL_SHADER_STORAGE_BUFFER,
				sizeInBytes,
				NULL, GL_STATIC_DRAW);

			glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

		});
	});

	Scheduler::Block();
}

OpenGLBuffer::~OpenGLBuffer() {

}