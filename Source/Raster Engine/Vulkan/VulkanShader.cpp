#include "VulkanShader.h"

#include <stdexcept>
#include <fstream>
#include <iostream>
#include <string>
#include <cassert>
#include <sstream>

/*
 *    Constructor.
 *    @param	filePath  	Full pathname of the file.
 *    @param	shaderType	Type of the shader.
 */

VulkanShader::VulkanShader(std::string filePath, shader_t shaderType)
	: Shader(filePath, shaderType) {

}
/* Destructor. */
VulkanShader::~VulkanShader() {

}