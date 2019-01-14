#pragma once

#include "Transput/Resource/Default/SPIRVLoader.h"
#include "Composition/Entity/EntityManager.h"

#include <vulkan/vulkan.hpp>

class VulkanShader {

public:

	VulkanShader(EntityManager&, Entity, const std::string&);
	virtual ~VulkanShader();

	VulkanShader(const VulkanShader&) = delete;
	VulkanShader(VulkanShader&& o) noexcept = delete;

	VulkanShader& operator=(const VulkanShader&) = delete;
	VulkanShader& operator=(VulkanShader&& other) noexcept = delete;

	vk::ShaderModule CreateModule(const vk::Device&, const std::string&);

	virtual vk::PipelineShaderStageCreateInfo CreateInfo() = 0;

protected:

	vk::ShaderModule module_;

private:

	EntityManager & entityManager_;
	Entity device_;

};
