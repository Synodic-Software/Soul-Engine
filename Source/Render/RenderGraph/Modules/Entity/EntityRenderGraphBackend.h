#pragma once

#include "Render/RenderGraph/RenderGraphModule.h"

class EntityRenderGraphBackend final : public RenderGraphModule {

public:

	EntityRenderGraphBackend(std::shared_ptr<RasterModule>&, std::shared_ptr<SchedulerModule>&);
	~EntityRenderGraphBackend() override = default;

	EntityRenderGraphBackend(const EntityRenderGraphBackend &) = delete;
	EntityRenderGraphBackend(EntityRenderGraphBackend &&) noexcept = default;

	EntityRenderGraphBackend& operator=(const EntityRenderGraphBackend &) = delete;
	EntityRenderGraphBackend& operator=(EntityRenderGraphBackend &&) noexcept = default;


	void Execute() override;

	void CreatePass(std::string,
		std::function<std::function<void(const Graph&, CommandList&)>(Graph&)>) override;

private:

	Graph renderGraph_;
	std::vector<std::function<void(const Graph&, CommandList&)>> graphTasks_;

};