//---------------------------------------------------------------------------------------------------
//@file	N:\Documents\Soul Engine\Source Files\Raster Engine\Vulkan\VulkanShader.h.
//Declares the vulkan shader class.

#pragma once

#include <string>

#include "Raster Engine\Shader.h"

//A vulkan shader.
class VulkanShader:public Shader {
public:

	//---------------------------------------------------------------------------------------------------
	//Constructor.
	//@param	parameter1	The first parameter.
	//@param	parameter2	The second parameter.

	VulkanShader(std::string, shader_t);
	//Destructor.
	virtual ~VulkanShader();
private:

};