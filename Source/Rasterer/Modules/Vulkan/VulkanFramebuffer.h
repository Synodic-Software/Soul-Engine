//#pragma once
//
//#include "Composition/Entity/Entity.h"
//#include "Composition/Component/Component.h"
//
//#include <vulkan/vulkan.hpp>
//#include <glm/vec2.hpp>
//
//class VulkanRenderPass;
//class EntityManager;
//
//class VulkanFramebuffer : Component<VulkanFramebuffer> {
//
//public:
//
//	VulkanFramebuffer(EntityManager&, Entity, vk::ImageView& swapChainImageView, VulkanRenderPass&, glm::uvec2&);
//	~VulkanFramebuffer() override;
//
//	VulkanFramebuffer(const VulkanFramebuffer&) = delete;
//	VulkanFramebuffer(VulkanFramebuffer&& o) noexcept = default;
//
//	VulkanFramebuffer& operator=(const VulkanFramebuffer&) = delete;
//	VulkanFramebuffer& operator=(VulkanFramebuffer&& other) noexcept = default;
//
//
//	const vk::Framebuffer& GetFrameBuffer() const;
//
//private:
//	
//	vk::Framebuffer frameBuffer_;
//
//	EntityManager* entityManager_;
//	Entity device_;
//
//};
