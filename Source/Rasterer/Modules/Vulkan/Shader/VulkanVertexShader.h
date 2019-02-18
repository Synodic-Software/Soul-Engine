#pragma once

#include "Rasterer/Modules/Vulkan/VulkanShader.h"

#include <string>
#include <vulkan/vulkan.hpp>

class VulkanVertexShader : public VulkanShader{

public:

	VulkanVertexShader(std::shared_ptr<VulkanDevice>&, const std::string&);
	virtual ~VulkanVertexShader() = default;

	VulkanVertexShader(const VulkanVertexShader&) = delete;
	VulkanVertexShader(VulkanVertexShader&& o) noexcept = delete;

	VulkanVertexShader& operator=(const VulkanVertexShader&) = delete;
	VulkanVertexShader& operator=(VulkanVertexShader&& other) noexcept = delete;

	vk::PipelineShaderStageCreateInfo CreateInfo();

};
