#pragma once

#include <string>
#include "Raster Engine/Buffer.h"

#include "Metrics.h"

class OpenGLBuffer : public Buffer {
public:

	OpenGLBuffer(uint sizeInBytes);

	~OpenGLBuffer();

	uint GetBufferID() {
		return buffer;
	}

protected:


private:
	uint buffer;

};