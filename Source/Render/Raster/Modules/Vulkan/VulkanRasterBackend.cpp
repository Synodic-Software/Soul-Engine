#include "VulkanRasterBackend.h"

#include "Device/VulkanDevice.h"
#include "Device/VulkanPhysicalDevice.h"
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
	entityRegistry_(entityRegistry),
	validationLayers_ {"VK_LAYER_KHRONOS_validation"}
{

	// setup Vulkan app info
	vk::ApplicationInfo appInfo;
	appInfo.apiVersion = VK_API_VERSION_1_1;
	appInfo.applicationVersion =
		VK_MAKE_VERSION(1, 0, 0);  // TODO forward the application version here
	appInfo.pApplicationName = "Soul Engine";  // TODO forward the application name here
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);  // TODO forward the engine version here
	appInfo.pEngineName = "Soul Engine";  // TODO forward the engine name here

	// The display will forward the extensions needed for Vulkan
	const auto newExtensions = windowModule_->GetRasterExtensions();

	requiredInstanceExtensions_.insert(
		std::end(requiredInstanceExtensions_), std::begin(newExtensions), std::end(newExtensions));

	// TODO minimize memory/runtime impact
	if constexpr (Compiler::Debug()) {

		requiredInstanceExtensions_.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

		std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();

		for (auto layer : validationLayers_) {

			bool found = false;
			for (const auto& layerProperties : availableLayers) {

				if (strcmp(layer, layerProperties.layerName) == 0) {
					found = true;
					break;
				}
			}

			if (!found) {

				throw std::runtime_error("Specified Vulkan validation layer is not available.");
			}
		}
	}

	vk::InstanceCreateInfo instanceCreationInfo;
	instanceCreationInfo.pApplicationInfo = &appInfo;
	instanceCreationInfo.enabledExtensionCount =
		static_cast<uint32>(requiredInstanceExtensions_.size());
	instanceCreationInfo.ppEnabledExtensionNames = requiredInstanceExtensions_.data();

	if constexpr (Compiler::Debug()) {

		instanceCreationInfo.enabledLayerCount = static_cast<uint32>(validationLayers_.size());
		instanceCreationInfo.ppEnabledLayerNames = validationLayers_.data();
	}
	else {

		instanceCreationInfo.enabledLayerCount = 0;
	}

	instance_ = createInstance(instanceCreationInfo);


	// Create debugging callback
	if constexpr (Compiler::Debug()) {

		dispatcher_ = vk::DispatchLoaderDynamic(instance_);

		vk::DebugUtilsMessengerCreateInfoEXT messengerCreateInfo;
		messengerCreateInfo.flags = vk::DebugUtilsMessengerCreateFlagBitsEXT(0);
		messengerCreateInfo.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
											  vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
		messengerCreateInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
										  vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
										  vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
		messengerCreateInfo.pfnUserCallback = DebugCallback;
		messengerCreateInfo.pUserData = nullptr;

		debugMessenger_ =
			instance_.createDebugUtilsMessengerEXT(messengerCreateInfo, nullptr, dispatcher_);
	}


	// setup devices
	auto physicalDevices = instance_.enumeratePhysicalDevices();
	physicalDevices_.reserve(physicalDevices.size());
	/*devices_.reserve(physicalDevices.size());
	commandPools_.reserve(physicalDevices.size());*/

	for (auto& physicalDevice : physicalDevices) {

		physicalDevices_.emplace_back(instance_, physicalDevice);

	}

	//devices_.push_back(std::make_shared<VulkanDevice>(scheduler, physicalDevice));
	//commandPools_.push_back(std::make_shared<VulkanCommandPool>(scheduler, devices_.back()));

}

VulkanRasterBackend::~VulkanRasterBackend()
{

	for (auto& device : devices_) {

		device->Synchronize();
	}

	devices_.clear();

	if constexpr (Compiler::Debug()) {

		instance_.destroyDebugUtilsMessengerEXT(debugMessenger_, nullptr, dispatcher_);
	}

	instance_.destroy();
}

void VulkanRasterBackend::Render()
{
	auto& device = devices_[0];

	for (const auto& [id, swapchain] : swapChains_) {

		 swapchain->Present(*commandBuffers_[0]);

	}
}

Entity VulkanRasterBackend::CreatePass(Entity swapchainID)
{
	// TODO: multiple devices
	auto& device = devices_[0];

	Entity renderPassID = entityRegistry_->CreateEntity();

	renderPassSwapchainMap_[renderPassID] = swapchainID;
	auto& swapchain = swapChains_[swapchainID];

	renderPasses_[renderPassID] =
		std::make_unique<VulkanRenderPass>(device, swapchain->GetFormat());

	//TODO: Better management if commandBuffers
	renderPassCommands_[renderPassID] = commandBuffers_.emplace_back(
		std::make_unique<VulkanCommandBuffer>(commandPools_.back(), devices_.back(),
			vk::CommandBufferUsageFlagBits::eSimultaneousUse, vk::CommandBufferLevel::ePrimary)).get();

	//TODO: Better management of pipelines
	pipelines_[renderPassID] = std::make_unique<VulkanPipeline>(device, swapchain->GetSize(),
		Resource("../Resources/Shaders/vert.spv"), Resource("../Resources/Shaders/frag.spv"),
		swapchain->GetFormat());


	vk::ImageViewCreateInfo colorAttachmentCreateInfo;
	colorAttachmentCreateInfo.format = format_;
	colorAttachmentCreateInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
	colorAttachmentCreateInfo.subresourceRange.levelCount = 1;
	colorAttachmentCreateInfo.subresourceRange.layerCount = 1;
	colorAttachmentCreateInfo.viewType = vk::ImageViewType::e2D;

	images_.resize(swapChainImages.size());
	for (uint32_t i = 0; i < swapChainImages.size(); ++i) {
		images_[i].image = swapChainImages[i];
		colorAttachmentCreateInfo.image = swapChainImages[i];
		images_[i].view = logicalDevice.createImageView(colorAttachmentCreateInfo);
		images_[i].fence = vk::Fence();
	}

	auto& frameBuffers = renderPassBuffers_[renderPassID];
	frameBuffers.reserve(frameBufferSize_);
	for (SwapChainImage& image : images_) {
		frameBuffers_.emplace_back(vkDevice_, image.view, pipeline_->GetRenderPass(), size);
	}

	for (const auto& image : images_) {
		logicalDevice.destroyImageView(image.view);
	}


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


	auto& swapchain = swapChains_[renderPassSwapchainMap_[renderpassID]];
	auto& frameBuffers = renderPassBuffers_[renderpassID];
	auto& renderPass = renderPasses_[renderpassID];
	auto& commandBuffer = renderPassCommands_[renderpassID]->Get();


	vk::RenderPassBeginInfo renderPassInfo;
	renderPassInfo.renderPass = renderPass->Get();
	renderPassInfo.framebuffer = frameBuffers[i].Get();
	renderPassInfo.renderArea.offset = vk::Offset2D(0, 0);
	renderPassInfo.renderArea.extent = {swapchain->GetSize().x, swapchain->GetSize().y};
	renderPassInfo.clearValueCount = 1;
	renderPassInfo.pClearValues = &clearColor;

	commandBuffer.begin(beginInfo);
	commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);



	commandBuffer.endRenderPass();
	commandBuffer.end();

}

Entity VulkanRasterBackend::RegisterSurface(std::any anySurface, glm::uvec2 size)
{

	auto surface = std::any_cast<vk::SurfaceKHR>(anySurface);

	// TODO: multiple devices
	auto& device = devices_[0];
	const vk::PhysicalDevice& physicalDevice = device->GetPhysical();

	if (!physicalDevice.getSurfaceSupportKHR(device->GetGraphicsIndex(), surface)) {

		throw NotImplemented();
	}

	const auto format = device->GetSurfaceFormat(surface);

	Entity surfaceID = entityRegistry_->CreateEntity();

	surfaces_[surfaceID] = surface;
	swapChains_[surfaceID] = std::make_unique<VulkanSwapChain>(
		device, surface, format.colorFormat, format.colorSpace, size, false);


	return surfaceID;
}

void VulkanRasterBackend::UpdateSurface(Entity surfaceID, glm::uvec2 size)
{

	auto surface = surfaces_[surfaceID];

	auto& device = devices_[0];
	const vk::PhysicalDevice& physicalDevice = device->GetPhysical();

	if (!physicalDevice.getSurfaceSupportKHR(device->GetGraphicsIndex(), surface)) {

		throw NotImplemented();
	}

	const auto format = device->GetSurfaceFormat(surface);

	auto newSwapchain = std::make_unique<VulkanSwapChain>(device, surface, format.colorFormat,
		format.colorSpace, size, false, swapChains_[surfaceID].get());

	swapChains_[surfaceID].swap(newSwapchain);
}

void VulkanRasterBackend::RemoveSurface(Entity surfaceID)
{

	swapChains_.erase(surfaceID);

	instance_.destroySurfaceKHR(surfaces_[surfaceID]);
	surfaces_.erase(surfaceID);
}

void VulkanRasterBackend::Compile(CommandList&)
{


}

void VulkanRasterBackend::Draw(DrawCommand& command, vk::CommandBuffer& commandBuffer)
{

	vk::Rect2D scissorRect;

	scissorRect.offset.x = command.scissorOffset.x;
	scissorRect.offset.y = command.scissorOffset.y;
	scissorRect.extent.width = command.scissorExtent.x;
	scissorRect.extent.height = command.scissorExtent.y;

	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline_->GetPipeline());

	vk::Buffer vertexBuffers[] = { command.vertexBuffer
	};
	vk::DeviceSize offsets[] = {
		0
	};

	commandBuffer.bindVertexBuffers(0, 1, vertexBuffers, offsets);
	commandBuffer.bindIndexBuffer(command.indexBuffer, 0, vk::IndexType::eUint16);


	commandBuffer.setScissor(0, 1, &scissorRect);
	commandBuffer.drawIndexed(command.elementSize, 1, command.indexOffset, command.vertexOffset, 0);

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

vk::Instance& VulkanRasterBackend::GetInstance()
{
	return instance_;
}

VkBool32 VulkanRasterBackend::DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData)
{

	throw NotImplemented();
}
