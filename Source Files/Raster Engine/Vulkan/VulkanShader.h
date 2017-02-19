#pragma once

#include <string>

#include "Raster Engine\Shader.h"

class VulkanShader:public Shader {
public:
	VulkanShader(std::string, shader_t);
	virtual ~VulkanShader();
private:

};