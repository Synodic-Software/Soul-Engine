#pragma once

#include "Rasterer/Graphics API/Shader/FragmentShader.h"
#include "Rasterer/Graphics API/Vulkan/VulkanShader.h"

class VulkanFragmentShader : public FragmentShader, public VulkanShader {

public:

	VulkanFragmentShader(EntityManager&, Entity, const std::string&);
	virtual ~VulkanFragmentShader() = default;

	VulkanFragmentShader(const VulkanFragmentShader&) = delete;
	VulkanFragmentShader(VulkanFragmentShader&& o) noexcept = delete;

	VulkanFragmentShader& operator=(const VulkanFragmentShader&) = delete;
	VulkanFragmentShader& operator=(VulkanFragmentShader&& other) noexcept = delete;

	vk::PipelineShaderStageCreateInfo CreateInfo() override;

};
