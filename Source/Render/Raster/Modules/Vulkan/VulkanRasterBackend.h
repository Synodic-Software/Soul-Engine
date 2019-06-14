#pragma once

#include "Render/Raster/RasterModule.h"

#include "VulkanSwapChain.h"
#include "Command/VulkanCommandPool.h"
#include "Command/VulkanCommandBuffer.h"

#include <vulkan/vulkan.hpp>
#include <glm/vec2.hpp>
#include <unordered_map>

class SchedulerModule;
class WindowModule;
class VulkanDevice;
class VulkanSwapChain;
class VulkanPhysicalDevice;

class VulkanRasterBackend final : public RasterModule {

public:

	VulkanRasterBackend(std::shared_ptr<SchedulerModule>&,
		std::shared_ptr<EntityRegistry>&,
		std::shared_ptr<WindowModule>&);
	~VulkanRasterBackend() override;

	VulkanRasterBackend(const VulkanRasterBackend &) = delete;
	VulkanRasterBackend(VulkanRasterBackend &&) noexcept = default;

	VulkanRasterBackend& operator=(const VulkanRasterBackend &) = delete;
	VulkanRasterBackend& operator=(VulkanRasterBackend &&) noexcept = default;

	void Render() override;
	Entity CreatePass(Entity) override;
	Entity CreateSubPass(Entity) override;
	void ExecutePass(Entity, CommandList&) override;

	Entity RegisterSurface(std::any, glm::uvec2) override;
	void UpdateSurface(Entity, glm::uvec2) override;
	void RemoveSurface(Entity) override;

	/*
	 * Simplifies a commandlist into a ready-to-execute format. Creates the opportunity for commandList reuse
	 *
	 * @param [in,out]	commandList	The commandList to simplify.
	 */

	void Compile(CommandList& commandList) override;


	vk::Instance& GetInstance();



private:

	std::shared_ptr<EntityRegistry> entityRegistry_;

	void Draw(DrawCommand&, vk::CommandBuffer&);
	void DrawIndirect(DrawIndirectCommand&, vk::CommandBuffer&);
	void UpdateBuffer(UpdateBufferCommand&, vk::CommandBuffer&);
	void UpdateTexture(UpdateTextureCommand&, vk::CommandBuffer&);
	void CopyBuffer(CopyBufferCommand&, vk::CommandBuffer&);
	void CopyTexture(CopyTextureCommand&, vk::CommandBuffer&);

	std::vector<std::shared_ptr<VulkanCommandPool>> commandPools_;
	std::vector<std::unique_ptr<VulkanCommandBuffer>> commandBuffers_;

	std::unordered_map<Entity, vk::SurfaceKHR> surfaces_;
	std::unordered_map<Entity, std::unique_ptr<VulkanSwapChain>> swapChains_;


	std::unordered_map<Entity, Entity> renderPassSwapchainMap_;
	std::unordered_map<Entity, std::unique_ptr<VulkanPipeline>> pipelines_;
	std::unordered_map<Entity, std::unique_ptr<VulkanRenderPass>> renderPasses_;
	std::unordered_map<Entity, VulkanCommandBuffer* > renderPassCommands_;
	std::unordered_map<Entity, std::vector<VulkanFrameBuffer>> renderPassBuffers_;

	std::vector<char const*> requiredInstanceExtensions_;
	std::vector<const char*> validationLayers_;

	vk::Instance instance_;
	std::vector<VulkanPhysicalDevice> devices_;
	std::vector<std::shared_ptr<VulkanDevice>> physicalDevices_;

	//Dynamic dispatcher for extensions
	vk::DispatchLoaderDynamic dispatcher_;

	//Debug state 
	//TODO: Should be conditionally included when the class is only debug mode.

	static VkBool32 DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT*, void*);

	vk::DebugUtilsMessengerEXT debugMessenger_;
	vk::DebugReportCallbackEXT callback_;

};