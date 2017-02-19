#pragma once

#include <string>

enum shader_t {VERTEX_SHADER,FRAGMENT_SHADER};

class Shader {
public:

	Shader(std::string, shader_t);
	virtual ~Shader();
	void ExtractShader(const std::string&);

protected:

	std::string name;
	std::string codeStr;
	shader_t type;
	unsigned* referenceCount;
	void Retain();
	void Release();

private:
	

};