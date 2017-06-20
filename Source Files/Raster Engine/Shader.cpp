#include "Shader.h"

#include <stdexcept>
#include <fstream>
#include <iostream>
#include <string>
#include <cassert>
#include <sstream>

#include "Utility\Logger.h"

Shader::Shader(std::string filePath, shader_t shaderType) :
	name(filePath), type(shaderType), referenceCount(nullptr){
	ExtractShader(filePath);

	referenceCount = new unsigned;
	*referenceCount = 1;
}
Shader::~Shader() {
	if (referenceCount) Release();
}

void Shader::ExtractShader(const std::string& filePath) {

	//Open shader file
	std::ifstream file;
	file.open(filePath.c_str(), std::ios::in | std::ios::binary);
	if (!file.is_open()) {
		S_LOG_ERROR("Failed to open file: ", filePath);
	}

	//read whole file into stringstream buffer
	std::stringstream buffer;
	buffer << file.rdbuf();

	//update code
	codeStr =buffer.str();
}
void Shader::Retain() {
	assert(referenceCount);
	*referenceCount += 1;
}

void Shader::Release() {
	assert(referenceCount && *referenceCount > 0);
	*referenceCount -= 1;
	if (*referenceCount == 0) {
		delete referenceCount; referenceCount = nullptr;
		delete this;	
	}
}