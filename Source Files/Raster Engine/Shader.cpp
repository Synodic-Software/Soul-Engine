#include "Shader.h"

#include <stdexcept>
#include <fstream>
#include <iostream>
#include <string>
#include <cassert>
#include <sstream>

Shader::Shader(const std::string& shaderCode, std::string filePath, shader_t shaderType) :
	name(filePath) {

}

Shader Shader::ExtractShader(const std::string& filePath, shader_t shaderType) {

	//Open shader file
	std::ifstream file;
	file.open(filePath.c_str(), std::ios::in | std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error(std::string("Failed to open file: ") + filePath);
	}

	//read whole file into stringstream buffer
	std::stringstream buffer;
	buffer << file.rdbuf();

	//return new shader
	Shader shader(buffer.str(), filePath, shaderType);
	return shader;
}