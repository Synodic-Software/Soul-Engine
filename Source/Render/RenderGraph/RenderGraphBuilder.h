#pragma once

class RenderGraphBuilder{

public:

	RenderGraphBuilder() = default;
	virtual ~RenderGraphBuilder() = default;

	RenderGraphBuilder(const RenderGraphBuilder &) = delete;
	RenderGraphBuilder(RenderGraphBuilder &&) noexcept = default;

	RenderGraphBuilder& operator=(const RenderGraphBuilder &) = delete;
	RenderGraphBuilder& operator=(RenderGraphBuilder &&) noexcept = default;


};
