#include "VulkanRasterBackend.h"

#include "Device/VulkanDevice.h"
#include "Core/Utility/Exception/Exception.h"
#include "Core/System/Compiler.h"
#include "Core/Composition/Entity/EntityRegistry.h"
#include "Types.h"
#include "Display/Window/WindowModule.h"
#include "VulkanSwapChain.h"
#include "Render/Raster/CommandList.h"
#include "Transput/Resource/Resource.h"

VulkanRasterBackend::VulkanRasterBackend(std::shared_ptr<SchedulerModule>& scheduler,
	std::shared_ptr<EntityRegistry>& entityRegistry,
	std::shared_ptr<WindowModule>& windowModule_):
	entityRegistry_(entityRegistry)
{

	// setup Vulkan app info
	vk::ApplicationInfo appInfo;
	appInfo.apiVersion = VK_API_VERSION_1_1;
	appInfo.applicationVersion =
		VK_MAKE_VERSION(1, 0, 0);  // TODO forward the application version here
	appInfo.pApplicationName = "Soul Engine";  // TODO forward the application name here
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);  // TODO forward the engine version here
	appInfo.pEngineName = "Soul Engine";  // TODO forward the engine name here


	std::vector<std::string> validationLayers;
	std::vector<std::string> instanceExtensions {
		VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
		VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME
	};

	// The display will forward the extensions needed for Vulkan
	const auto windowExtensions = windowModule_->GetRasterExtensions();

	instanceExtensions.insert(
		std::end(instanceExtensions), std::begin(windowExtensions), std::end(windowExtensions));

	if constexpr (Compiler::Debug()) {

		validationLayers.push_back("VK_LAYER_KHRONOS_validation");
		instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

	}

	instance_.reset(new VulkanInstance(appInfo, validationLayers, instanceExtensions));

	std::vector<std::string> deviceExtensions {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, 
		VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME
	};

	physicalDevices_ = instance_->EnumeratePhysicalDevices();
	device_ = std::make_unique<VulkanDevice>(scheduler, instance_->Handle(), physicalDevices_[0].Handle(), validationLayers, deviceExtensions);
}

void VulkanRasterBackend::Present()
{

	throw NotImplemented();
}

Entity VulkanRasterBackend::CreatePass(Entity swapchainID)
{

	Entity renderPassID = entityRegistry_->CreateEntity();

	renderPassSwapchainMap_[renderPassID] = swapchainID;
	// auto& swapchain = swapChains_.at(swapchainID);

	// renderPasses_.try_emplace(renderPassID, device_->GetLogical());

	// TODO: Better management if commandBuffers
	/*renderPassCommands_[renderPassID] =
		commandBuffers_
			.emplace_back(std::make_unique<VulkanCommandBuffer>(commandPools_.back(),
				device_->GetLogical(),
				vk::CommandBufferUsageFlagBits::eSimultaneousUse,
				vk::CommandBufferLevel::ePrimary))
			.get();*/

	// TODO: Better management of pipelines
	// pipelines_[renderPassID] = std::make_unique<VulkanPipeline>(device_, swapchain->GetSize(),
	//	Resource("../Resources/Shaders/vert.spv"), Resource("../Resources/Shaders/frag.spv"),
	//	swapchain->GetFormat());


	// vk::ImageViewCreateInfo colorAttachmentCreateInfo;
	// colorAttachmentCreateInfo.format = format_;
	// colorAttachmentCreateInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
	// colorAttachmentCreateInfo.subresourceRange.levelCount = 1;
	// colorAttachmentCreateInfo.subresourceRange.layerCount = 1;
	// colorAttachmentCreateInfo.viewType = vk::ImageViewType::e2D;

	// images_.resize(swapChainImages.size());
	// for (uint32_t i = 0; i < swapChainImages.size(); ++i) {
	//	images_[i].image = swapChainImages[i];
	//	colorAttachmentCreateInfo.image = swapChainImages[i];
	//	images_[i].view = logicalDevice.createImageView(colorAttachmentCreateInfo);
	//	images_[i].fence = vk::Fence();
	//}

	// auto& frameBuffers = renderPassBuffers_[renderPassID];
	// frameBuffers.reserve(frameBufferSize_);
	// for (SwapChainImage& image : images_) {
	//	frameBuffers_.emplace_back(vkDevice_, image.view, pipeline_->GetRenderPass(), size);
	//}

	// for (const auto& image : images_) {
	//	logicalDevice.destroyImageView(image.view);
	//}


	return renderPassID;
}

Entity VulkanRasterBackend::CreateSubPass(Entity parentPass)
{

	Entity subPassID = entityRegistry_->CreateEntity();

	return subPassID;
}

void VulkanRasterBackend::ExecutePass(Entity renderpassID, CommandList& commandList)
{

	vk::ClearValue clearColor(vk::ClearColorValue(std::array<float, 4> {0.0f, 0.0f, 0.0f, 1.0f}));

	vk::CommandBufferBeginInfo beginInfo;
	beginInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;
	beginInfo.pInheritanceInfo = nullptr;


	auto& swapchain = swapChains_.at(renderPassSwapchainMap_.at(renderpassID));
	auto& frameBuffers = renderPassBuffers_.at(renderpassID);
	auto& renderPass = renderPasses_.at(renderpassID);
	auto& commandBuffer = renderPassCommands_.at(renderpassID)->Get();


	vk::RenderPassBeginInfo renderPassInfo;
	renderPassInfo.renderPass = renderPass.Handle();
	// renderPassInfo.framebuffer = frameBuffers[i].Handle();
	renderPassInfo.renderArea.offset = vk::Offset2D(0, 0);
	renderPassInfo.renderArea.extent = swapchain.GetSize();
	renderPassInfo.clearValueCount = 1;
	renderPassInfo.pClearValues = &clearColor;

	commandBuffer.begin(beginInfo);
	commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);


	commandBuffer.endRenderPass();
	commandBuffer.end();
}

Entity VulkanRasterBackend::RegisterSurface(std::any anySurface, glm::uvec2 size)
{

	auto surfaceHandle = std::any_cast<vk::SurfaceKHR>(anySurface);
	Entity surfaceID = entityRegistry_->CreateEntity();

	auto& [surfaceIterator, didInsert] =
		surfaces_.try_emplace(surfaceID, instance_->Handle(), surfaceHandle);

	VulkanSurface& surface = surfaceIterator->second;
	const auto format = surface.UpdateFormat(*device_);
	swapChains_.try_emplace(surfaceID, device_, surface, false);

	return surfaceID;
}

void VulkanRasterBackend::UpdateSurface(Entity surfaceID, glm::uvec2 size)
{

	auto& surface = surfaces_.at(surfaceID);
	const auto format = surface.UpdateFormat(*device_);

	auto& oldSwapchain = swapChains_.at(surfaceID);
	auto& newSwapchain = VulkanSwapChain(device_, surface, false, &oldSwapchain);

	std::swap(oldSwapchain, newSwapchain);
}

void VulkanRasterBackend::RemoveSurface(Entity surfaceID)
{

	swapChains_.erase(surfaceID);
	surfaces_.erase(surfaceID);
}

void VulkanRasterBackend::Compile(CommandList&)
{
}

VulkanInstance& VulkanRasterBackend::GetInstance()
{

	return *instance_;
}

void VulkanRasterBackend::Draw(DrawCommand& command, vk::CommandBuffer& commandBuffer)
{

	/*vk::Rect2D scissorRect;

	scissorRect.offset.x = command.scissorOffset.x;
	scissorRect.offset.y = command.scissorOffset.y;
	scissorRect.extent.width = command.scissorExtent.x;
	scissorRect.extent.height = command.scissorExtent.y;

	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline_->GetPipeline());

	vk::Buffer vertexBuffers[] = {command.vertexBuffer};
	vk::DeviceSize offsets[] = {0};

	commandBuffer.bindVertexBuffers(0, 1, vertexBuffers, offsets);
	commandBuffer.bindIndexBuffer(command.indexBuffer, 0, vk::IndexType::eUint16);


	commandBuffer.setScissor(0, 1, &scissorRect);
	commandBuffer.drawIndexed(command.elementSize, 1, command.indexOffset, command.vertexOffset,
	0);*/
}

void VulkanRasterBackend::DrawIndirect(DrawIndirectCommand&, vk::CommandBuffer& commandBuffer)
{

	throw NotImplemented();
}
void VulkanRasterBackend::UpdateBuffer(UpdateBufferCommand&, vk::CommandBuffer& commandBuffer)
{
}
void VulkanRasterBackend::UpdateTexture(UpdateTextureCommand&, vk::CommandBuffer& commandBuffer)
{

	throw NotImplemented();
}
void VulkanRasterBackend::CopyBuffer(CopyBufferCommand&, vk::CommandBuffer& commandBuffer)
{

	throw NotImplemented();
}
void VulkanRasterBackend::CopyTexture(CopyTextureCommand&, vk::CommandBuffer& commandBuffer)
{

	throw NotImplemented();
}
