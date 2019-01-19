#pragma once

#include "Core/Interface/Module/Module.h"

#include <memory>

class RasterDevice : public Module {

public:

	RasterDevice() = default;
	virtual ~RasterDevice() = default;

	RasterDevice(const RasterDevice &) = delete;
	RasterDevice(RasterDevice &&) noexcept = default;

	RasterDevice& operator=(const RasterDevice &) = delete;
	RasterDevice& operator=(RasterDevice &&) noexcept = default;

	virtual void Synchronize() = 0;

	//Factory
	static std::unique_ptr<RasterDevice> CreateModule();

};
