#pragma once

#include "Core/Interface/Module/Module.h"

#include <memory>

class RenderGraphModule : public Module<RenderGraphModule> {

public:

	RenderGraphModule() = default;
	virtual ~RenderGraphModule() = default;

	RenderGraphModule(const RenderGraphModule &) = delete;
	RenderGraphModule(RenderGraphModule &&) noexcept = default;

	RenderGraphModule& operator=(const RenderGraphModule &) = delete;
	RenderGraphModule& operator=(RenderGraphModule &&) noexcept = default;


	// Factory
	static std::shared_ptr<RenderGraphModule> CreateModule();

};
