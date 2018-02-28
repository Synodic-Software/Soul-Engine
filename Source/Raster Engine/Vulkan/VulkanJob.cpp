#include "VulkanJob.h"

#include <stdexcept>
#include <fstream>
#include <iostream>
#include <string>
#include <cassert>
#include <sstream>

/* Default constructor. */
VulkanJob::VulkanJob()
	: RasterJob() {

}
/* Destructor. */
VulkanJob::~VulkanJob() {

}

/*
 *    Attach shaders.
 *    @param	parameter1	The first parameter.
 */

void VulkanJob::AttachShaders(const std::vector<Shader*>&) {

}

/*
 *    Registers the uniform described by uniformName.
 *    @param	uniformName	Name of the uniform.
 */

void VulkanJob::RegisterUniform(const std::string uniformName) {

}

/*
 *    Uploads a geometry.
 *    @param [in,out]	parameter1	If non-null, the first parameter.
 *    @param 		 	parameter2	The second parameter.
 *    @param [in,out]	parameter3	If non-null, the third parameter.
 *    @param 		 	parameter4	The fourth parameter.
 */

void VulkanJob::UploadGeometry(float*, uint, uint*, uint) {

}

/*
 *    Sets an uniform.
 *    @param	std::string	The standard string.
 *    @param	parameter2 	The second parameter.
 */

void VulkanJob::SetUniform(const std::string, RasterVariant) {

}

/* Draws this object. */
void VulkanJob::Draw() {

}