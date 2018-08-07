#include "VulkanVertexShader.h"

VulkanVertexShader::VulkanVertexShader(EntityManager& entityManger, Entity device, const std::string& fileName) :
	Shader(fileName),
	VertexShader(fileName),
	VulkanShader(entityManger, device, fileName)
{
}

vk::PipelineShaderStageCreateInfo VulkanVertexShader::CreateInfo() {

	vk::PipelineShaderStageCreateInfo vertShaderStageInfo;
	vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
	vertShaderStageInfo.module = module_;
	vertShaderStageInfo.pName = "main";

	return vertShaderStageInfo;

}
