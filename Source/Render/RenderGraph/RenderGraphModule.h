#pragma once

#include "Core/Interface/Module/Module.h"

#include "CommandList.h"

#include <memory>
#include <string>
#include <functional>

class RasterModule;

class RenderGraphModule : public Module<RenderGraphModule> {

public:

	RenderGraphModule() = default;
	virtual ~RenderGraphModule() = default;

	RenderGraphModule(const RenderGraphModule &) = delete;
	RenderGraphModule(RenderGraphModule &&) noexcept = default;

	RenderGraphModule& operator=(const RenderGraphModule &) = delete;
	RenderGraphModule& operator=(RenderGraphModule &&) noexcept = default;


	virtual void CreatePass(std::string, std::function<std::function<void(CommandList&)>()>) = 0;

	// Factory
	static std::shared_ptr<RenderGraphModule> CreateModule(std::shared_ptr<RasterModule>&);


};
