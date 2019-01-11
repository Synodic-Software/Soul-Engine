#pragma once

#include "Core/Interface/Module/Module.h"

#include <memory>

class RasterDevice;

class RasterBackend : public Module {

public:

	RasterBackend() = default;
	virtual ~RasterBackend() = default;

	RasterBackend(const RasterBackend &) = delete;
	RasterBackend(RasterBackend &&) noexcept = default;

	RasterBackend& operator=(const RasterBackend &) = delete;
	RasterBackend& operator=(RasterBackend &&) noexcept = default;

	virtual void Draw() = 0;
	virtual void DrawIndirect() = 0;

	virtual std::shared_ptr<RasterDevice> CreateDevice() = 0;

	//Factory
	static std::shared_ptr<RasterBackend> CreateModule();

};
