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

	VulkanRenderPass& GetRenderPass();
	const vk::Pipeline& GetPipeline() const;

    //TODO: temporary
	const VulkanBuffer<Vertex>& GetVertexBuffer() const;
	const VulkanBuffer<uint16>& GetIndexBuffer() const;


private:

	std::shared_ptr<VulkanDevice> device_;
	VulkanRenderPass renderPass_;

	VulkanShader vertexShader_;
	VulkanShader fragmentShader_;

	vk::PipelineLayout pipelineLayout_;
	vk::Pipeline pipeline_;
	vk::PipelineCache pipelineCache_;

    //TODO: temporary, refactor
    VulkanBuffer<Vertex> vertexBuffer_;
	VulkanBuffer<Vertex> vertexStagingBuffer_;

    VulkanBuffer<uint16> indexBuffer_;
	VulkanBuffer<uint16> indexStagingBuffer_;

};
