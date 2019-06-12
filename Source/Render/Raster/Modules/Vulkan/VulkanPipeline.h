#pragma once

#include "VulkanRenderPass.h"
#include "VulkanShader.h"
#include "Buffer/VulkanBuffer.h"

#include <vulkan/vulkan.hpp>

class Resource;
class VulkanDevice;
class Vertex;

class VulkanPipeline {

public:

	VulkanPipeline(std::shared_ptr<VulkanDevice>&, vk::Extent2D&, const Resource&, const Resource&, vk::Format);
	~VulkanPipeline();

	VulkanPipeline(const VulkanPipeline&) = delete;
	VulkanPipeline(VulkanPipeline&&) noexcept = delete;

	VulkanPipeline& operator=(const VulkanPipeline&) = delete;
	VulkanPipeline& operator=(VulkanPipeline&&) noexcept = delete;

	const vk::Pipeline& GetPipeline() const;


private:

	std::shared_ptr<VulkanDevice> device_;

	std::vector<VulkanShader> stages_;

	vk::PipelineLayout pipelineLayout_;
	vk::Pipeline pipeline_;
	vk::PipelineCache pipelineCache_;


};
