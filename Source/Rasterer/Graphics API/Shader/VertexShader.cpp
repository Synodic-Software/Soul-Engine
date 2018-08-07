#include "VertexShader.h"

VertexShader::VertexShader(const std::string& fileName) :
	Shader(fileName)
{
	Shader::shaderType_ = ShaderType::Vertex;
}
