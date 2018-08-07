#include "VulkanShader.h"
#include "VulkanDevice.h"

VulkanShader::VulkanShader(EntityManager& entityManger, Entity device, const std::string& fileName) :
	Shader(fileName),
	entityManager_(entityManger),
	device_(device)
{
	const auto& vkDevice = entityManager_.GetComponent<VulkanDevice>(device_);
	const vk::Device& logicalDevice = vkDevice.GetLogicalDevice();

	module_ = CreateModule(logicalDevice, fileName);

}

VulkanShader::~VulkanShader() {

	const auto& vkDevice = entityManager_.GetComponent<VulkanDevice>(device_);
	const vk::Device& logicalDevice = vkDevice.GetLogicalDevice();

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

