#pragma once

#include "Rasterer/Graphics API/Shader/VertexShader.h"
#include "Rasterer/Graphics API/Vulkan/VulkanShader.h"

class VulkanVertexShader : public VertexShader, public VulkanShader {

public:

	VulkanVertexShader(EntityManager&, Entity, const std::string&);
	virtual ~VulkanVertexShader() = default;

	VulkanVertexShader(const VulkanVertexShader&) = delete;
	VulkanVertexShader(VulkanVertexShader&& o) noexcept = delete;

	VulkanVertexShader& operator=(const VulkanVertexShader&) = delete;
	VulkanVertexShader& operator=(VulkanVertexShader&& other) noexcept = delete;

	vk::PipelineShaderStageCreateInfo CreateInfo() override;

};
