//#pragma once
//
//#include "Composition/Entity/EntityManager.h"
//#include "VulkanRenderPass.h"
//#include "Shader/VulkanVertexShader.h"
//#include "Shader/VulkanFragmentShader.h"
//
//#include <vulkan/vulkan.hpp>
//
//#include <string_view>
//
//
//class VulkanDevice;
//
//class VulkanPipeline {
//
//public:
//
//	VulkanPipeline(VulkanDevice&, vk::Extent2D&, const std::string&, const std::string&, vk::Format);
//	~VulkanPipeline();
//
//	VulkanPipeline(const VulkanPipeline&) = delete;
//	VulkanPipeline(VulkanPipeline&&) noexcept = delete;
//
//	VulkanPipeline& operator=(const VulkanPipeline&) = delete;
//	VulkanPipeline& operator=(VulkanPipeline&&) noexcept = delete;
//
//	void Create(VulkanDevice&, vk::Extent2D&, vk::Format);
//
//	VulkanRenderPass& GetRenderPass();
//	const vk::Pipeline& GetPipeline() const;
//
//private:
//
//	VulkanRenderPass renderPass_;
//
//	VulkanVertexShader vertexShader_;
//	VulkanFragmentShader fragmentShader_;
//
//	vk::PipelineLayout pipelineLayout_;
//	vk::Pipeline pipeline_;
//
//};
