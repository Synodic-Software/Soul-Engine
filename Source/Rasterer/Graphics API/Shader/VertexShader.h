#pragma once

#include "Rasterer/Graphics API/Shader.h"

class VertexShader : public virtual Shader {

public:

	VertexShader(const std::string&);
	virtual ~VertexShader() = default;

	VertexShader(const VertexShader&) = delete;
	VertexShader(VertexShader&& o) noexcept = delete;

	VertexShader& operator=(const VertexShader&) = delete;
	VertexShader& operator=(VertexShader&& other) noexcept = delete;

};
