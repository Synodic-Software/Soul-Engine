//---------------------------------------------------------------------------------------------------
//@file	N:\Documents\Soul Engine\Source Files\Raster Engine\OpenGL\OpenGLBuffer.h.
//Declares the open gl buffer class.

#pragma once

#include <string>
#include "Raster Engine/Buffer.h"

#include "Metrics.h"

//Buffer for open gl.
class OpenGLBuffer : public Buffer {
public:

	//---------------------------------------------------------------------------------------------------
	//Constructor.
	//@param	sizeInBytes	The size in bytes.

	OpenGLBuffer(uint sizeInBytes);

	//Destructor.
	~OpenGLBuffer();

	//---------------------------------------------------------------------------------------------------
	//Gets buffer identifier.
	//@return	The buffer identifier.

	uint GetBufferID() {
		return buffer;
	}

protected:


private:
	//The buffer
	uint buffer;

};