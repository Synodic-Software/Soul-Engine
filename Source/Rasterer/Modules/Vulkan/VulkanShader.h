#pragma once

#include <vulkan/vulkan.hpp>

#include <filesystem>


class Resource;
class VulkanDevice;

class VulkanShader {

public:

	VulkanShader(std::shared_ptr<VulkanDevice>&, const vk::ShaderStageFlagBits&, const Resource&);
	~VulkanShader();

	VulkanShader(const VulkanShader&) = delete;
	VulkanShader(VulkanShader&& o) noexcept = delete;

	VulkanShader& operator=(const VulkanShader&) = delete;
	VulkanShader& operator=(VulkanShader&& other) noexcept = delete;

	vk::ShaderModule CreateModule(const vk::Device&, const std::filesystem::path&);

	vk::PipelineShaderStageCreateInfo GetInfo();


private:

	vk::ShaderModule module_;
	vk::PipelineShaderStageCreateInfo info_;

	std::shared_ptr<VulkanDevice>& device_;

};
