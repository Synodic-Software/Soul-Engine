#include "VulkanShader.h"
#include "VulkanDevice.h"

VulkanShader::VulkanShader(std::shared_ptr<VulkanDevice>& device, const std::string& fileName) :
	device_(device)
{
	const vk::Device& logicalDevice = device_->GetLogical();

	module_ = CreateModule(logicalDevice, fileName);

}

VulkanShader::~VulkanShader() {

	const vk::Device& logicalDevice = device_->GetLogical();

	logicalDevice.destroyShaderModule(module_, nullptr);

}

vk::ShaderModule VulkanShader::CreateModule(const vk::Device& device, const std::string& fileName) {

	//TODO: abstract into some sort of loader
	std::ifstream file(fileName, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		assert(false);
	}

	const size_t fileSize = static_cast<size_t>(file.tellg());
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);

	file.close();

	vk::ShaderModuleCreateInfo createInfo;
	createInfo.codeSize = buffer.size();
	createInfo.pCode = reinterpret_cast<const uint32_t*>(buffer.data());

	return device.createShaderModule(createInfo, nullptr);

}

