#pragma once

#include "Composition/Entity/EntityManager.h"

#include <vulkan/vulkan.hpp>

class VulkanRenderPass
{

public:

	VulkanRenderPass(EntityManager&, Entity, vk::Format);
	~VulkanRenderPass();

	VulkanRenderPass(const VulkanRenderPass&) = delete;
	VulkanRenderPass(VulkanRenderPass&&) noexcept = delete;

	VulkanRenderPass& operator=(const VulkanRenderPass&) = delete;
	VulkanRenderPass& operator=(VulkanRenderPass&&) noexcept = delete;

	void Create(vk::Format);
	const vk::RenderPass& GetRenderPass() const;

private:

	EntityManager & entityManager_;
	Entity device_;
	vk::RenderPass renderPass_;

};
