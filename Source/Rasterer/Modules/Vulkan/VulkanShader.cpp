#include "VulkanShader.h"
#include "VulkanDevice.h"
#include "Core/Utility/Exception/Exception.h"
#include "Transput/Resource/Resource.h"

#include <fstream>


VulkanShader::VulkanShader(std::shared_ptr<VulkanDevice>& device, const vk::ShaderStageFlagBits& shaderType, const Resource& resource) :
	device_(device)
{
	const vk::Device& logicalDevice = device_->GetLogical();

	module_ = CreateModule(logicalDevice, resource.Path());

    info_.stage = shaderType;
	info_.module = module_;
	info_.pName = "main";

}

VulkanShader::~VulkanShader() {

	const vk::Device& logicalDevice = device_->GetLogical();

	logicalDevice.destroyShaderModule(module_, nullptr);

}

vk::ShaderModule VulkanShader::CreateModule(const vk::Device& device, const std::filesystem::path& path) const {

	if (!std::filesystem::exists(path)) {
		throw NotImplemented();
	}

	//TODO: abstract into some sort of loader
	std::ifstream file(path.c_str(), std::ifstream::ate | std::ios::binary);

	const size_t fileSize = static_cast<size_t>(file.tellg());
	std::vector<std::byte> buffer(fileSize);

	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
	file.close();

	vk::ShaderModuleCreateInfo createInfo;
	createInfo.codeSize = buffer.size();
	createInfo.pCode = reinterpret_cast<const uint32_t*>(buffer.data());

	return device.createShaderModule(createInfo, nullptr);

}

vk::PipelineShaderStageCreateInfo VulkanShader::GetInfo() const {

	return info_;

}
