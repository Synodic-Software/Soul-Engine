#include "VulkanPipeline.h"

#include "VulkanDevice.h"

VulkanPipeline::VulkanPipeline(EntityManager& entityManger, Entity device, vk::Extent2D& extent, const std::string& vertexFilename, const std::string& fragmentFilename, vk::Format swapChainFormat) :
	entityManager_(entityManger),
	device_(device),
	renderPass_(entityManager_, device_, swapChainFormat), //TODO: remove hardcoded renderpass + allow multiple renderpasses
	vertexShader_(entityManger, device, vertexFilename),
	fragmentShader_(entityManger, device, fragmentFilename)
{

	Create(extent, swapChainFormat, false);

}

void VulkanPipeline::Create(vk::Extent2D& extent, vk::Format swapChainFormat, bool createRenderPass) {

	if (createRenderPass) {
		renderPass_.Create(swapChainFormat);
	}

	//SETUP LAYOUT
	vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
	pipelineLayoutInfo.setLayoutCount = 0;
	pipelineLayoutInfo.pushConstantRangeCount = 0;

	const auto& vkDevice = entityManager_.GetComponent<VulkanDevice>(device_);
	const vk::Device& logicalDevice = vkDevice.GetLogicalDevice();

	pipelineLayout_ = logicalDevice.createPipelineLayout(pipelineLayoutInfo, nullptr);


	vk::PipelineShaderStageCreateInfo shaderStages[] = {
		vertexShader_.CreateInfo(),
		fragmentShader_.CreateInfo()
	};

	vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
	vertexInputInfo.vertexBindingDescriptionCount = 0;
	vertexInputInfo.vertexAttributeDescriptionCount = 0;

	vk::PipelineInputAssemblyStateCreateInfo inputAssembly;
	inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
	inputAssembly.primitiveRestartEnable = VK_FALSE;

	vk::Viewport viewport;
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = extent.width;
	viewport.height = extent.height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	vk::Rect2D scissor;
	scissor.offset = vk::Offset2D(0, 0);
	scissor.extent = extent;

	vk::PipelineViewportStateCreateInfo viewportState;
	viewportState.viewportCount = 1;
	viewportState.pViewports = &viewport;
	viewportState.scissorCount = 1;
	viewportState.pScissors = &scissor;

	vk::PipelineRasterizationStateCreateInfo rasterizer;
	rasterizer.polygonMode = vk::PolygonMode::eFill;
	rasterizer.lineWidth = 1.0f;
	rasterizer.cullMode = vk::CullModeFlagBits::eBack;
	rasterizer.frontFace = vk::FrontFace::eClockwise;
	rasterizer.depthBiasEnable = VK_FALSE;
	rasterizer.depthClampEnable = VK_FALSE;
	rasterizer.rasterizerDiscardEnable = VK_FALSE;

	vk::PipelineMultisampleStateCreateInfo multisampling;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;

	vk::PipelineColorBlendAttachmentState colorBlendAttachment;
	colorBlendAttachment.colorWriteMask =
		vk::ColorComponentFlagBits::eR |
		vk::ColorComponentFlagBits::eG |
		vk::ColorComponentFlagBits::eB |
		vk::ColorComponentFlagBits::eA;
	colorBlendAttachment.blendEnable = VK_FALSE;

	vk::PipelineColorBlendStateCreateInfo colorBlending;
	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.logicOp = vk::LogicOp::eCopy;
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &colorBlendAttachment;
	colorBlending.blendConstants[0] = 0.0f;
	colorBlending.blendConstants[1] = 0.0f;
	colorBlending.blendConstants[2] = 0.0f;
	colorBlending.blendConstants[3] = 0.0f;

	vk::PipelineDepthStencilStateCreateInfo depthStencil;
	depthStencil.depthTestEnable = VK_TRUE;
	depthStencil.depthWriteEnable = VK_TRUE;
	depthStencil.depthCompareOp = vk::CompareOp::eLessOrEqual;
	depthStencil.depthBoundsTestEnable = VK_FALSE;
	depthStencil.back.failOp = vk::StencilOp::eKeep;
	depthStencil.back.passOp = vk::StencilOp::eKeep;
	depthStencil.back.compareOp = vk::CompareOp::eAlways;
	depthStencil.stencilTestEnable = VK_FALSE;
	depthStencil.front = depthStencil.back;

	vk::GraphicsPipelineCreateInfo pipelineInfo;
	pipelineInfo.stageCount = 2;
	pipelineInfo.pStages = shaderStages;
	pipelineInfo.pVertexInputState = &vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState = &multisampling;
	pipelineInfo.pDepthStencilState = &depthStencil;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.layout = pipelineLayout_;
	pipelineInfo.renderPass = renderPass_.GetRenderPass();


	pipeline_ = logicalDevice.createGraphicsPipeline(vkDevice.GetPipelineCache(), pipelineInfo);

}

void VulkanPipeline::Terminate() {

	const auto& vkDevice = entityManager_.GetComponent<VulkanDevice>(device_);
	const vk::Device& logicalDevice = vkDevice.GetLogicalDevice();

	logicalDevice.destroyPipeline(pipeline_);
	logicalDevice.destroyPipelineLayout(pipelineLayout_);

	logicalDevice.destroyRenderPass(renderPass_.GetRenderPass());

}

VulkanPipeline::~VulkanPipeline() {

	Terminate();

}

VulkanRenderPass& VulkanPipeline::GetRenderPass() {
	return renderPass_;
}

const vk::Pipeline& VulkanPipeline::GetPipeline() const {
	return pipeline_;
}
