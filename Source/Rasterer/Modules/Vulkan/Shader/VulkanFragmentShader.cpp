//#include "VulkanFragmentShader.h"
//
//VulkanFragmentShader::VulkanFragmentShader(EntityManager& entityManger, Entity device, const std::string& fileName) :
//	VulkanShader(entityManger, device,fileName)
//{
//}
//
//vk::PipelineShaderStageCreateInfo VulkanFragmentShader::CreateInfo() {
//
//	vk::PipelineShaderStageCreateInfo fragShaderStageInfo;
//	fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
//	fragShaderStageInfo.module = module_;
//	fragShaderStageInfo.pName = "main";
//
//	return fragShaderStageInfo;
//
//}
