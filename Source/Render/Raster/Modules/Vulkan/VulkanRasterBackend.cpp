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
	entityRegistry_(entityRegistry),
	currentFrame_(0)
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
		VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME};

	// The display will forward the extensions needed for Vulkan
	const auto windowExtensions = windowModule_->GetRasterExtensions();

	instanceExtensions.insert(
		std::end(instanceExtensions), std::begin(windowExtensions), std::end(windowExtensions));

	if constexpr (Compiler::Debug()) {

		validationLayers.push_back("VK_LAYER_KHRONOS_validation");
		instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}

	instance_.reset(new VulkanInstance(appInfo, validationLayers, instanceExtensions));

	std::vector<std::string> deviceExtensions {VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
		VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME};

	// TODO: Device groups and multiple devices
	physicalDevices_ = instance_->EnumeratePhysicalDevices();
	devices_.emplace_back(scheduler, instance_->Handle(), physicalDevices_[0].Handle(),
		validationLayers, deviceExtensions);

	commandPools_.reserve(frameMax_ * devices_.size());

	// Set up synchronization primitives
	vk::SemaphoreCreateInfo semaphoreInfo;

	vk::FenceCreateInfo fenceInfo;
	fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

	for (int i = 0; i < frameMax_; ++i) {
		for (auto& vkDevice : devices_) {

			auto& device = vkDevice.Logical();
			commandPools_.emplace_back(scheduler, vkDevice);

			presentSemaphores_[i] = device.createSemaphore(semaphoreInfo);
			renderSemaphores_[i] = device.createSemaphore(semaphoreInfo);
			frameFences_[i] = device.createFence(fenceInfo);
		}
	}
}

VulkanRasterBackend::~VulkanRasterBackend()
{

	for (size_t i = 0; i < frameMax_; i++) {
		for (auto& vkDevice : devices_) {

			auto& device = vkDevice.Logical();
			device.destroySemaphore(presentSemaphores_[i]);
			device.destroySemaphore(renderSemaphores_[i]);
			device.destroyFence(frameFences_[i]);
		}
	}
}

void VulkanRasterBackend::Present()
{

	throw NotImplemented();
}

Entity VulkanRasterBackend::CreatePass(std::function<void(Entity)> function)
{

	// Create new entity
	Entity renderPassID = entityRegistry_->CreateEntity();

	// TODO: real resource
	Entity resource = entityRegistry_->CreateEntity();

	// Create empty pass data
	renderPassAttachments_.try_emplace(renderPassID);
	renderPassSubPasses_.try_emplace(renderPassID);
	renderPassDependencies_.try_emplace(renderPassID);

	// Default subpass and default output
	CreatePassOutput(renderPassID, resource, Format::RGBA);

	CreateSubPass(renderPassID, [&](Entity subPassID) { function(subPassID); });

	std::vector<vk::AttachmentDescription2KHR>& subPassAttachments =
		renderPassAttachments_.at(renderPassID);
	std::vector<vk::SubpassDescription2KHR>& subPassDescriptions =
		renderPassSubPasses_.at(renderPassID);
	std::vector<vk::SubpassDependency2KHR>& subPassDependencies =
		renderPassDependencies_.at(renderPassID);

	renderPasses_.try_emplace(
		renderPassID, devices_[0], subPassAttachments, subPassDescriptions, subPassDependencies);

	// TODO: command buffer per thread per ID
	renderPassCommandBuffers_.try_emplace(renderPassID, commandPools_.back().Handle(),
		devices_[0].Logical(), vk::CommandBufferUsageFlagBits::eSimultaneousUse,
		vk::CommandBufferLevel::ePrimary);


	// TODO: Better management of pipelines
	// pipelines_[renderPassID] = std::make_unique<VulkanPipeline>(device_, swapchain->GetSize(),
	//	Resource("../Resources/Shaders/vert.spv"), Resource("../Resources/Shaders/frag.spv"),
	//	swapchain->GetFormat());

	return renderPassID;
}

Entity VulkanRasterBackend::CreateSubPass(Entity renderPassID, std::function<void(Entity)> function)
{

	Entity subPassID = entityRegistry_->CreateEntity();

	subPassAttachmentReferences_.try_emplace(subPassID);

	function(subPassID);

	auto& outputAttachmentReferences = subPassAttachmentReferences_.at(subPassID);

	// Create the subpass object
	vk::SubpassDescription2KHR subPass;
	subPass.flags = vk::SubpassDescriptionFlags();
	subPass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
	subPass.viewMask = 0;
	subPass.inputAttachmentCount = 0;
	subPass.pInputAttachments = nullptr;
	subPass.colorAttachmentCount = outputAttachmentReferences.size();
	subPass.pColorAttachments = outputAttachmentReferences.data();
	subPass.pResolveAttachments = nullptr;
	subPass.pDepthStencilAttachment = nullptr;
	subPass.preserveAttachmentCount = 0;
	subPass.pPreserveAttachments = nullptr;

	// push the subpass object to the parent renderPass
	renderPassSubPasses_.try_emplace(renderPassID);
	renderPassSubPasses_.at(renderPassID).push_back(subPass);

	return subPassID;
}

void VulkanRasterBackend::ExecutePass(Entity renderPassID,
	Entity surfaceID,
	CommandList& commandList)
{

	vk::ClearValue clearColor(vk::ClearColorValue(std::array<float, 4> {0.0f, 0.0f, 0.0f, 1.0f}));

	auto& swapchain = swapChains_.at(surfaceID);

	auto& renderPass = renderPasses_.at(renderPassID);
	auto& commandBuffer = renderPassCommandBuffers_.at(renderPassID);
	auto& commandBufferHandle = commandBuffer.Handle();

	//TODO: Cleanup this mess
	// Generate the framebuffer attachments
	if (!frameBuffers_.at(surfaceID).has_value()) {

		std::array<VulkanFrameBuffer, frameMax_> frameBuffers = {
			VulkanFrameBuffer {
				devices_[0].Logical(), swapchain.ImageViews(), renderPass, swapchain.Size()},
			VulkanFrameBuffer {
				devices_[0].Logical(), swapchain.ImageViews(), renderPass, swapchain.Size()},
			VulkanFrameBuffer {
				devices_[0].Logical(), swapchain.ImageViews(), renderPass, swapchain.Size()}};

		frameBuffers_.at(surfaceID).emplace(std::move(frameBuffers));

	}
	else {

		for (auto& frameBuffer : frameBuffers_.at(surfaceID).value()) {
			frameBuffer = VulkanFrameBuffer(
				devices_[0].Logical(), swapchain.ImageViews(), renderPass, swapchain.Size());
		}

	}

	auto& frameBuffers = frameBuffers_.at(surfaceID).value();

	vk::RenderPassBeginInfo renderPassInfo;
	renderPassInfo.renderPass = renderPass.Handle();
	renderPassInfo.framebuffer = frameBuffers[currentFrame_].Handle();
	renderPassInfo.renderArea.offset = vk::Offset2D(0, 0);
	renderPassInfo.renderArea.extent = swapchain.Size();
	renderPassInfo.clearValueCount = 1;
	renderPassInfo.pClearValues = &clearColor;

	vk::SubpassBeginInfoKHR subPassInfo;
	subPassInfo.contents = vk::SubpassContents::eInline;

	commandBuffer.Begin();
	commandBufferHandle.beginRenderPass2KHR(
		renderPassInfo, subPassInfo, devices_[0].DispatchLoader());


	commandBufferHandle.endRenderPass();
	commandBuffer.End();
}

void VulkanRasterBackend::CreatePassInput(Entity passID, Entity resource, Format format)
{

	throw NotImplemented();
}

void VulkanRasterBackend::CreatePassOutput(Entity passID, Entity resource, Format format)
{

	std::vector<vk::AttachmentDescription2KHR>& renderPassAttachments =
		renderPassAttachments_.at(passID);

	uint attachmentIndex = renderPassAttachments.size();

	vk::AttachmentDescription2KHR& attachment = renderPassAttachments.emplace_back();
	attachment.flags = vk::AttachmentDescriptionFlags();
	attachment.format = ConvertFormat(format);
	attachment.samples = vk::SampleCountFlagBits::e1;
	attachment.loadOp = vk::AttachmentLoadOp::eClear;
	attachment.storeOp = vk::AttachmentStoreOp::eStore;
	attachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
	attachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
	attachment.initialLayout = vk::ImageLayout::eUndefined;
	attachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

	vk::AttachmentReference2KHR colorAttachmentRef;
	colorAttachmentRef.attachment = attachmentIndex;
	colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;
	colorAttachmentRef.aspectMask = vk::ImageAspectFlags();
}

Entity VulkanRasterBackend::CreateSurface(std::any anySurface, glm::uvec2 size)
{

	auto surfaceHandle = std::any_cast<vk::SurfaceKHR>(anySurface);
	Entity surfaceID = entityRegistry_->CreateEntity();

	auto& [surfaceIterator, didInsert] =
		surfaces_.try_emplace(surfaceID, instance_->Handle(), surfaceHandle);

	// Every surface needs a swapchain
	VulkanSurface& surface = surfaceIterator->second;
	const auto format = surface.UpdateFormat(devices_[0]);
	swapChains_.try_emplace(surfaceID, devices_[0], surface, false);
	auto& swapChain = swapChains_.at(surfaceID);

	// create the framebuffer storage for the surface
	frameBuffers_.try_emplace(surfaceID);

	return surfaceID;
}

void VulkanRasterBackend::UpdateSurface(Entity surfaceID, glm::uvec2 size)
{

	auto& surface = surfaces_.at(surfaceID);
	const auto format = surface.UpdateFormat(devices_[0]);

	auto& oldSwapchain = swapChains_.at(surfaceID);
	auto& newSwapchain = VulkanSwapChain(devices_[0], surface, false, &oldSwapchain);

	std::swap(oldSwapchain, newSwapchain);
}

void VulkanRasterBackend::RemoveSurface(Entity surfaceID)
{

	swapChains_.erase(surfaceID);
	surfaces_.erase(surfaceID);
}

void VulkanRasterBackend::AttachSurface(Entity renderPassID)
{
}

void VulkanRasterBackend::DetatchSurface(Entity renderPassID)
{
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


vk::Format VulkanRasterBackend::ConvertFormat(Format format)
{
	switch (format) {
		case Format::RGBA:
			return vk::Format::eR8G8B8A8Srgb;
		default:
			throw NotImplemented();
			return vk::Format::eUndefined;
	}
}
