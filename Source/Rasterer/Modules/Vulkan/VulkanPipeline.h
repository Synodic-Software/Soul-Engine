#pragma once

#include "Composition/Entity/EntityManager.h"
#include "VulkanRenderPass.h"
#include "Shader/VulkanVertexShader.h"
#include "Shader/VulkanFragmentShader.h"

#include <vulkan/vulkan.hpp>

#include <string_view>


class VulkanDevice;

class VulkanPipeline {

public:

	VulkanPipeline(std::shared_ptr<VulkanDevice>&, vk::Extent2D&, const std::string&, const std::string&, vk::Format);
	~VulkanPipeline();

	VulkanPipeline(const VulkanPipeline&) = delete;
	VulkanPipeline(VulkanPipeline&&) noexcept = delete;

	VulkanPipeline& operator=(const VulkanPipeline&) = delete;
	VulkanPipeline& operator=(VulkanPipeline&&) noexcept = delete;

	VulkanRenderPass& GetRenderPass();
	const vk::Pipeline& GetPipeline() const;

private:

	std::shared_ptr<VulkanDevice> device_;
	VulkanRenderPass renderPass_;

	VulkanVertexShader vertexShader_;
	VulkanFragmentShader fragmentShader_;

	vk::PipelineLayout pipelineLayout_;
	vk::Pipeline pipeline_;
	vk::PipelineCache pipelineCache_;

};
