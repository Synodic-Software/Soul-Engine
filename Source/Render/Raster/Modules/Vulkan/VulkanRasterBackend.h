#pragma once

#include "Render/Raster/RasterModule.h"

#include "VulkanSwapChain.h"
#include "VulkanSurface.h"
#include "Command/VulkanCommandPool.h"
#include "Command/VulkanCommandBuffer.h"
#include "Device/VulkanPhysicalDevice.h"
#include "VulkanInstance.h"

#include <vulkan/vulkan.hpp>
#include <glm/vec2.hpp>
#include <unordered_map>
#include <memory>

class SchedulerModule;
class WindowModule;
class VulkanDevice;
class VulkanSwapChain;

class VulkanRasterBackend final : public RasterModule {

public:

	VulkanRasterBackend(std::shared_ptr<SchedulerModule>&,
		std::shared_ptr<EntityRegistry>&,
		std::shared_ptr<WindowModule>&);
	~VulkanRasterBackend() override = default;

	VulkanRasterBackend(const VulkanRasterBackend &) = delete;
	VulkanRasterBackend(VulkanRasterBackend &&) noexcept = default;

	VulkanRasterBackend& operator=(const VulkanRasterBackend &) = delete;
	VulkanRasterBackend& operator=(VulkanRasterBackend &&) noexcept = default;

	void Present() override;

	//RenderPass Management
	Entity CreatePass(std::function<void(Entity)>) override;
	Entity CreateSubPass(std::function<void(Entity)>) override;
	void ExecutePass(Entity, CommandList&) override;

	//RenderPass Modification
	void CreatePassInput(Entity, Entity, Format) override;
	void CreatePassOutput(Entity, Entity, Format) override;

	Entity CreateSurface(std::any, glm::uvec2) override;
	void UpdateSurface(Entity, glm::uvec2) override;
	void RemoveSurface(Entity) override;

	/*
	 * Simplifies a commandlist into a ready-to-execute format. Creates the opportunity for commandList reuse
	 *
	 * @param [in,out]	commandList	The commandList to simplify.
	 */

	void Compile(CommandList& commandList) override;

	VulkanInstance& GetInstance();

private:

	vk::Format ConvertFormat(Format);

	std::shared_ptr<EntityRegistry> entityRegistry_;

	void Draw(DrawCommand&, vk::CommandBuffer&);
	void DrawIndirect(DrawIndirectCommand&, vk::CommandBuffer&);
	void UpdateBuffer(UpdateBufferCommand&, vk::CommandBuffer&);
	void UpdateTexture(UpdateTextureCommand&, vk::CommandBuffer&);
	void CopyBuffer(CopyBufferCommand&, vk::CommandBuffer&);
	void CopyTexture(CopyTextureCommand&, vk::CommandBuffer&);

	std::vector<VulkanPhysicalDevice> physicalDevices_;
	std::unique_ptr<VulkanDevice> device_;

	std::unordered_map<Entity, VulkanSurface> surfaces_;
	std::unordered_map<Entity, VulkanSwapChain> swapChains_;

	std::vector<std::shared_ptr<VulkanCommandPool>> commandPools_;
	std::vector<std::unique_ptr<VulkanCommandBuffer>> commandBuffers_;

	std::unordered_map<Entity, Entity> renderPassSwapchainMap_;

	std::unordered_map<Entity, std::vector<vk::AttachmentDescription2KHR>> renderPassAttachments_;
	std::unordered_map<Entity, std::vector<vk::SubpassDescription2KHR>> renderPassSubPasses_;
	std::unordered_map<Entity, std::vector<vk::SubpassDependency2KHR>> renderPassDependencies_;
	//map of subpasses to list of attachment IDs
	std::unordered_map<Entity, std::vector<uint>> subPassAttachmentUses_;


	std::unordered_map<Entity, VulkanPipeline> pipelines_;
	std::unordered_map<Entity, VulkanRenderPass> renderPasses_;
	std::unordered_map<Entity, VulkanSubPass> subPasses_;
	std::unordered_map<Entity, VulkanCommandBuffer* > renderPassCommands_;
	std::unordered_map<Entity, VulkanFrameBuffer> renderPassBuffers_;
	
	//TODO: put on stack and remove deferred construction
	std::unique_ptr<VulkanInstance> instance_;


};