#include "OpenGLShader.h"

#include <stdexcept>
#include <fstream>
#include <iostream>
#include <string>
#include <cassert>
#include <sstream>

OpenGLShader::OpenGLShader(const std::string& shaderCode, std::string filePath, shader_t shaderType)
	: Shader(shaderCode, filePath, shaderType) {

}
