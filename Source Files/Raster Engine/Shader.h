#pragma once

#include <string>

enum shader_t {VERTEX_SHADER,FRAGMENT_SHADER};

class Shader {
public:
	const std::string& name;
	Shader(const std::string&, std::string, shader_t);
	Shader ExtractShader(const std::string&, shader_t);

private:

};