#include "VulkanFragmentShader.h"

VulkanFragmentShader::VulkanFragmentShader(std::shared_ptr<VulkanDevice>& device, const std::string& fileName) :
	VulkanShader(device, fileName)
{
}

vk::PipelineShaderStageCreateInfo VulkanFragmentShader::CreateInfo() {

	vk::PipelineShaderStageCreateInfo fragShaderStageInfo;
	fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
	fragShaderStageInfo.module = module_;
	fragShaderStageInfo.pName = "main";

	return fragShaderStageInfo;

}
