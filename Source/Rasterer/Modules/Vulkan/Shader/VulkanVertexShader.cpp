#include "VulkanVertexShader.h"

VulkanVertexShader::VulkanVertexShader(std::shared_ptr<VulkanDevice>& device, const std::string& fileName) :
	VulkanShader(device, fileName)
{
}

vk::PipelineShaderStageCreateInfo VulkanVertexShader::CreateInfo() {

	vk::PipelineShaderStageCreateInfo vertShaderStageInfo;
	vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
	vertShaderStageInfo.module = module_;
	vertShaderStageInfo.pName = "main";

	return vertShaderStageInfo;

}
