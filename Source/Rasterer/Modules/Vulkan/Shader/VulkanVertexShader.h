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
//class VulkanVertexShader : public VulkanShader{
//
//public:
//
//	VulkanVertexShader(EntityManager&, Entity, const std::string&);
//	virtual ~VulkanVertexShader() = default;
//
//	VulkanVertexShader(const VulkanVertexShader&) = delete;
//	VulkanVertexShader(VulkanVertexShader&& o) noexcept = delete;
//
//	VulkanVertexShader& operator=(const VulkanVertexShader&) = delete;
//	VulkanVertexShader& operator=(VulkanVertexShader&& other) noexcept = delete;
//
//	vk::PipelineShaderStageCreateInfo CreateInfo();
//
//};
