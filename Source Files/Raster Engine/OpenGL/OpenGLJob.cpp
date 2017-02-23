#include "OpenGLJob.h"

#include <stdexcept>
#include <fstream>
#include <iostream>
#include <string>
#include <cassert>
#include <sstream>

OpenGLJob::OpenGLJob()
	: RasterJob(){

}

OpenGLJob::~OpenGLJob() {

}

void OpenGLJob::AttachShaders(const std::vector<Shader*>&) {

}
