#pragma once

#include <vulkan/vulkan.hpp>

#include <filesystem>

class VulkanDevice;

class VulkanShader {

public:

	VulkanShader(std::shared_ptr<VulkanDevice>&, const std::filesystem::path&);
	virtual ~VulkanShader();

	VulkanShader(const VulkanShader&) = delete;
	VulkanShader(VulkanShader&& o) noexcept = delete;

	VulkanShader& operator=(const VulkanShader&) = delete;
	VulkanShader& operator=(VulkanShader&& other) noexcept = delete;

	vk::ShaderModule CreateModule(const vk::Device&, const std::filesystem::path&);

	virtual vk::PipelineShaderStageCreateInfo CreateInfo() = 0;

protected:

	vk::ShaderModule module_;

private:

	std::shared_ptr<VulkanDevice>& device_;

};
