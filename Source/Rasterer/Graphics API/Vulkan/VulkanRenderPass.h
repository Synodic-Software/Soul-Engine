#pragma once

#include "VulkanDevice.h"
#include "Composition/Entity/EntityManager.h"
#include "Rasterer/Graphics API/RenderPass.h"

#include <vulkan/vulkan.hpp>

class VulkanRenderPass : public RenderPass
{

public:

	VulkanRenderPass(EntityManager&, Entity, vk::Format);
	~VulkanRenderPass() override;

	VulkanRenderPass(const VulkanRenderPass&) = delete;
	VulkanRenderPass(VulkanRenderPass&& o) noexcept = delete;

	VulkanRenderPass& operator=(const VulkanRenderPass&) = delete;
	VulkanRenderPass& operator=(VulkanRenderPass&& other) noexcept = delete;

	const vk::RenderPass& GetRenderPass() const;

private:

	EntityManager & entityManager_;
	Entity device_;
	vk::RenderPass renderPass_;

};
