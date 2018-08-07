#pragma once

#include "Rasterer/Graphics API/Shader.h"

class FragmentShader : public virtual Shader {

public:

	FragmentShader(const std::string&);
	virtual ~FragmentShader() = default;

	FragmentShader(const FragmentShader&) = delete;
	FragmentShader(FragmentShader&& o) noexcept = delete;

	FragmentShader& operator=(const FragmentShader&) = delete;
	FragmentShader& operator=(FragmentShader&& other) noexcept = delete;

};
