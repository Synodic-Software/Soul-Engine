#include "VulkanRasterBackend.h"

#include "VulkanDevice.h"
#include "Core/Utility/Exception/Exception.h"
#include "Core/System/Compiler.h"
#include "Core/Composition/Entity/EntityRegistry.h"
#include "Types.h"
#include "Display/Window/WindowModule.h"
#include "VulkanSwapChain.h"
#include "Render/Raster/CommandList.h"

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
	devices_.reserve(physicalDevices.size());
	commandPools_.reserve(physicalDevices.size());

	for (auto& physicalDevice : physicalDevices) {

		devices_.push_back(std::make_shared<VulkanDevice>(scheduler, physicalDevice));
		commandPools_.push_back(std::make_shared<VulkanCommandPool>(scheduler, devices_.back()));
		commandBuffers_.push_back(std::make_shared<VulkanCommandBuffer>(commandPools_.back(),
			devices_.back(), vk::CommandBufferUsageFlagBits::eSimultaneousUse,
			vk::CommandBufferLevel::ePrimary));
	}
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

		 swapchain->Present(commandBuffers_[i]);

	}
}

Entity VulkanRasterBackend::CreatePass(Entity swapchainID)
{
	// TODO: multiple devices
	auto& device = devices_[0];

	Entity renderPassID = entityRegistry_->CreateEntity();

	renderPasses_[renderPassID] =
		std::make_unique<VulkanRenderPass>(device, swapChains_[swapchainID]->GetFormat());

	// TODO: Remove hardcoded pipeline + Hardcoded paths
	// TODO: Associate paths to Project/Executable
	// pipeline_ = std::make_unique<VulkanPipeline>(vkDevice_, swapchainSize,
	// Resource("../Resources/Shaders/vert.spv"), Resource("../Resources/Shaders/frag.spv"),
	// colorFormat);

	/*frameBuffers_.reserve(images_.size());
	for (SwapChainImage& image : images_) {
		frameBuffers_.emplace_back(vkDevice_, image.view, pipeline_->GetRenderPass(), size);
	}*/


	return renderPassID;
}

void VulkanRasterBackend::ExecutePass(Entity renderpass, CommandList& commandList)
{

	vk::ClearValue clearColor(vk::ClearColorValue(std::array<float, 4> {0.0f, 0.0f, 0.0f, 1.0f}));

	vk::CommandBufferBeginInfo beginInfo;
	beginInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;
	beginInfo.pInheritanceInfo = nullptr;




	vk::RenderPassBeginInfo renderPassInfo;
	renderPassInfo.renderPass = pipeline_->GetRenderPass().GetRenderPass();
	renderPassInfo.framebuffer = frameBuffers_[i].GetFrameBuffer();
	renderPassInfo.renderArea.offset = vk::Offset2D(0, 0);
	renderPassInfo.renderArea.extent = swapchainSize;
	renderPassInfo.clearValueCount = 1;
	renderPassInfo.pClearValues = &clearColor;

	commandBuffers_[i].begin(beginInfo);
	commandBuffers_[i].beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);



	commandBuffers_[i].endRenderPass();
	commandBuffers_[i].end();

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

void VulkanRasterBackend::Draw(DrawCommand& command)
{

	vk::Rect2D scissorRect;

	scissorRect.offset.x = command.scissorOffset.x;
	scissorRect.offset.y = command.scissorOffset.y;
	scissorRect.extent.width = command.scissorExtent.x;
	scissorRect.extent.height = command.scissorExtent.y;

	commandBuffers_[0]->Get().setScissor(0, 1, &scissorRect);
	commandBuffers_[0]->Get().drawIndexed(
		command.elementSize, 1, command.indexOffset, command.vertexOffset, 0);

	// TODO: refactor below
	commandBuffers_[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline_->GetPipeline());

	vk::Buffer vertexBuffers[] = {pipeline_->GetVertexBuffer().GetBuffer()};
	vk::DeviceSize offsets[] = {0};

	commandBuffers_[i].bindVertexBuffers(0, 1, vertexBuffers, offsets);
	commandBuffers_[i].bindIndexBuffer(
		pipeline_->GetIndexBuffer().GetBuffer(), 0, vk::IndexType::eUint16);

	commandBuffers_[i].drawIndexed(6, 1, 0, 0, 0);

}

void VulkanRasterBackend::DrawIndirect(DrawIndirectCommand&)
{

	throw NotImplemented();

}
void VulkanRasterBackend::UpdateBuffer(UpdateBufferCommand&)
{

}
void VulkanRasterBackend::UpdateTexture(UpdateTextureCommand&)
{

	throw NotImplemented();

}
void VulkanRasterBackend::CopyBuffer(CopyBufferCommand&)
{

	throw NotImplemented();

}
void VulkanRasterBackend::CopyTexture(CopyTextureCommand&)
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
