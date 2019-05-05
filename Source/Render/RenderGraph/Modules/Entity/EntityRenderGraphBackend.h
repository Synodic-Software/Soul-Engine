#pragma once

#include "Render/RenderGraph/RenderGraphModule.h"

class RasterModule;

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
		std::function<std::function<void(const EntityRegistry&, CommandList&)>(Graph&)>) override;

private:

	std::shared_ptr<RasterModule> rasterModule_;
	Graph renderGraph_;
	EntityRegistry registry_;
	std::vector<std::function<void(const EntityRegistry&, CommandList&)>> graphTasks_;

};