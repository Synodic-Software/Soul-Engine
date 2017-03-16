#include "VulkanShader.h"

#include <stdexcept>
#include <fstream>
#include <iostream>
#include <string>
#include <cassert>
#include <sstream>

VulkanShader::VulkanShader(std::string filePath, shader_t shaderType)
	: Shader(filePath, shaderType) {

}
VulkanShader::~VulkanShader() {

}