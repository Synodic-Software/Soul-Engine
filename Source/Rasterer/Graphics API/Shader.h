#pragma once

#include <string>

enum class ShaderType { Vertex, Fragment };

class Shader {

public:

	Shader(const std::string&);
	virtual ~Shader() = default;

	Shader(const Shader&) = delete;
	Shader(Shader&& o) noexcept = delete;

	Shader& operator=(const Shader&) = delete;
	Shader& operator=(Shader&& other) noexcept = delete;

protected:

	ShaderType shaderType_;
	std::string fileName_;

};
