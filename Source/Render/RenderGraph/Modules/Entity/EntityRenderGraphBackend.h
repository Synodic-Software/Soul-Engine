#pragma once

#include "Render/RenderGraph/RenderGraphModule.h"


class EntityRenderGraphBackend final : public RenderGraphModule {

public:

	EntityRenderGraphBackend(std::shared_ptr<RasterModule>&);
	~EntityRenderGraphBackend() override = default;

	EntityRenderGraphBackend(const EntityRenderGraphBackend &) = delete;
	EntityRenderGraphBackend(EntityRenderGraphBackend &&) noexcept = default;

	EntityRenderGraphBackend& operator=(const EntityRenderGraphBackend &) = delete;
	EntityRenderGraphBackend& operator=(EntityRenderGraphBackend &&) noexcept = default;


	void Execute() override;

	void CreatePass(std::string,
		std::function<std::function<void(EntityReader&, CommandList&)>(
			EntityWriter&)>) override;

private:

	std::shared_ptr<RasterModule> rasterModule_;

	EntityRegistry graphRegistry_;
	std::vector<std::function<void(EntityReader&, CommandList&)>> graphTasks_;

};