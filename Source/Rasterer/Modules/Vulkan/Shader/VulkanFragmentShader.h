//#pragma once
//
//#include "Rasterer/Modules/Vulkan/VulkanShader.h"
//
//#include <string>
//#include <vulkan/vulkan.hpp>
//
//class Entity;
//class EntityManager;
//
//class VulkanFragmentShader : public VulkanShader {
//
//public:
//
//	VulkanFragmentShader(EntityManager&, Entity, const std::string&);
//	virtual ~VulkanFragmentShader() = default;
//
//	VulkanFragmentShader(const VulkanFragmentShader&) = delete;
//	VulkanFragmentShader(VulkanFragmentShader&& o) noexcept = delete;
//
//	VulkanFragmentShader& operator=(const VulkanFragmentShader&) = delete;
//	VulkanFragmentShader& operator=(VulkanFragmentShader&& other) noexcept = delete;
//
//	vk::PipelineShaderStageCreateInfo CreateInfo();
//
//};
