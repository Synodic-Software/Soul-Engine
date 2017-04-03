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
void VulkanJob::RegisterUniform(const std::string uniformName) {

}

void VulkanJob::UploadGeometry(float*, uint, uint*, uint) {

}

void VulkanJob::SetUniform(const std::string, RasterVariant) {

}

void VulkanJob::Draw() {

}