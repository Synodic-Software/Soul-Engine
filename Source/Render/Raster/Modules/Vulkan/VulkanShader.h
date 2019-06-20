#pragma once

#include <vulkan/vulkan.hpp>

#include <filesystem>


class Resource;
class VulkanDevice;

class VulkanShader {

public:

	VulkanShader(const vk::Device&, const vk::ShaderStageFlagBits&, const Resource&);
	~VulkanShader();

	VulkanShader(const VulkanShader&) = delete;
	VulkanShader(VulkanShader&& o) noexcept = delete;

	VulkanShader& operator=(const VulkanShader&) = delete;
	VulkanShader& operator=(VulkanShader&& other) noexcept = delete;

	vk::ShaderModule CreateModule(const vk::Device&, const std::filesystem::path&) const;

	vk::PipelineShaderStageCreateInfo GetInfo() const;


private:

	vk::ShaderModule module_;
	vk::PipelineShaderStageCreateInfo info_;

	vk::Device device_;

};
