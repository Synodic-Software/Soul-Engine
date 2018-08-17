#pragma once

#include "Composition/Entity/EntityManager.h"
#include "VulkanRenderPass.h"
#include "Shader/VulkanVertexShader.h"
#include "Shader/VulkanFragmentShader.h"
#include "Rasterer/Graphics API/Pipeline.h"

#include <vulkan/vulkan.hpp>

#include <string_view>


class VulkanPipeline : public Pipeline {

public:

	VulkanPipeline(EntityManager&, Entity, vk::Extent2D&, const std::string&, const std::string&, vk::Format);
	~VulkanPipeline() override;

	VulkanPipeline(const VulkanPipeline&) = delete;
	VulkanPipeline(VulkanPipeline&& o) noexcept = delete;

	VulkanPipeline& operator=(const VulkanPipeline&) = delete;
	VulkanPipeline& operator=(VulkanPipeline&& other) noexcept = delete;

	VulkanRenderPass& GetRenderPass();
	const vk::Pipeline& GetPipeline() const;

private:

	EntityManager& entityManager_;
	Entity device_;
	VulkanRenderPass renderPass_;

	VulkanVertexShader vertexShader_;
	VulkanFragmentShader fragmentShader_;

	vk::PipelineLayout pipelineLayout_;
	vk::Pipeline pipeline_;

};
