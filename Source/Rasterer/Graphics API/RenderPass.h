#pragma once

class RenderPass {

public:

	RenderPass() = default;
	virtual ~RenderPass() = default;

	RenderPass(const RenderPass&) = delete;
	RenderPass(RenderPass&& o) noexcept = delete;

	RenderPass& operator=(const RenderPass&) = delete;
	RenderPass& operator=(RenderPass&& other) noexcept = delete;


};
