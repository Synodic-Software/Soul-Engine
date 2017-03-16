#include "VulkanJob.h"

#include <stdexcept>
#include <fstream>
#include <iostream>
#include <string>
#include <cassert>
#include <sstream>

VulkanJob::VulkanJob()
	: RasterJob() {

}
VulkanJob::~VulkanJob() {

}

void VulkanJob::AttachShaders(const std::vector<Shader*>&) {

}