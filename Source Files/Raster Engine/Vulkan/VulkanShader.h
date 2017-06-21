#pragma once

#include <string>

#include "Raster Engine\Shader.h"

/* A vulkan shader. */
/* A vulkan shader. */
class VulkanShader:public Shader {
public:

	/*
	 *    Constructor.
	 *
	 *    @param	parameter1	The first parameter.
	 *    @param	parameter2	The second parameter.
	 */

	VulkanShader(std::string, shader_t);
	/* Destructor. */
	/* Destructor. */
	virtual ~VulkanShader();
private:

};