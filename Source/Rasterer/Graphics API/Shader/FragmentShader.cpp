#include "FragmentShader.h"

FragmentShader::FragmentShader(const std::string& fileName) :
	Shader(fileName)
{
	Shader::shaderType_ = ShaderType::Fragment;
}
