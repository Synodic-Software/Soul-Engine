#include "VulkanPipeline.h"

#include "VulkanDevice.h"
#include "Command/VulkanCommandBuffer.h"

#include "Core/Geometry/Vertex.h"
#include "Buffer/VulkanBuffer.h"


VulkanPipeline::VulkanPipeline(std::shared_ptr<VulkanDevice>& device,
	vk::Extent2D& extent,
	const Resource& vertexResource,
	const Resource& fragmentResource,
	vk::Format swapChainFormat):
	device_(device),
	renderPass_(device_,
		swapChainFormat),  // TODO: remove hardcoded renderpass + allow multiple renderpasses
	vertexShader_(device, vk::ShaderStageFlagBits::eVertex, vertexResource),
	fragmentShader_(device, vk::ShaderStageFlagBits::eFragment, fragmentResource)
{

	vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
	pipelineLayoutInfo.setLayoutCount = 0;
	pipelineLayoutInfo.pushConstantRangeCount = 0;

	const vk::Device& logicalDevice = device_->GetLogical();

	pipelineLayout_ = logicalDevice.createPipelineLayout(pipelineLayoutInfo, nullptr);

	vk::PipelineShaderStageCreateInfo shaderStages[] = {
		vertexShader_.GetInfo(), fragmentShader_.GetInfo()};


	// TODO: Refactor and move vertex attribute and bindings.
	vk::VertexInputBindingDescription bindingDescription;
	bindingDescription.binding = 0;
	bindingDescription.stride = sizeof(Vertex);
	bindingDescription.inputRate = vk::VertexInputRate::eVertex;

	std::array<vk::VertexInputAttributeDescription, 1> attributeDescriptions;

	attributeDescriptions[0].binding = 0;
	attributeDescriptions[0].location = 0;
	attributeDescriptions[0].format = vk::Format::eR32G32B32Sfloat;
	attributeDescriptions[0].offset = offsetof(Vertex, position);  // TODO: C++23 Reflection



	vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
	vertexInputInfo.vertexBindingDescriptionCount = 1;
	vertexInputInfo.vertexAttributeDescriptionCount = attributeDescriptions.size();
	vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
	vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();


	vk::PipelineInputAssemblyStateCreateInfo inputAssembly;
	inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
	inputAssembly.primitiveRestartEnable = VK_FALSE;

	vk::Viewport viewport;
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = static_cast<float>(extent.width);
	viewport.height = static_cast<float>(extent.height);
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
		vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
		vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
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

	vk::PipelineCacheCreateInfo pipelineCreateInfo;
	// TODO: pipeline serialization n' such

	pipelineCache_ = logicalDevice.createPipelineCache(pipelineCreateInfo);
	pipeline_ = logicalDevice.createGraphicsPipeline(pipelineCache_, pipelineInfo);
}

VulkanPipeline::~VulkanPipeline()
{

	const vk::Device& logicalDevice = device_->GetLogical();

	logicalDevice.destroyPipelineCache(pipelineCache_);
	logicalDevice.destroyPipeline(pipeline_);
	logicalDevice.destroyPipelineLayout(pipelineLayout_);
}

VulkanRenderPass& VulkanPipeline::GetRenderPass()
{
	return renderPass_;
}

const vk::Pipeline& VulkanPipeline::GetPipeline() const
{
	return pipeline_;
}