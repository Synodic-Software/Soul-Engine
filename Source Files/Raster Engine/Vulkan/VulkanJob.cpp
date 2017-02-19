#include "VulkanJob.h"

#include <stdexcept>
#include <fstream>
#include <iostream>
#include <string>
#include <cassert>
#include <sstream>

VulkanJob::VulkanJob(const std::vector<Shader>& shaders)
	: RasterJob(shaders) {

}
VulkanJob::~VulkanJob() {

}