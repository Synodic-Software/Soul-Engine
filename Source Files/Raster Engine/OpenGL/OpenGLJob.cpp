#include "OpenGLJob.h"

#include <stdexcept>
#include <fstream>
#include <iostream>
#include <string>
#include <cassert>
#include <sstream>

OpenGLJob::OpenGLJob(const std::vector<Shader>& shaders)
	: RasterJob(shaders){

}

OpenGLJob::~OpenGLJob() {

}
