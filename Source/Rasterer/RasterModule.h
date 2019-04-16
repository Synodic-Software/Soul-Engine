#pragma once

#include "Core/Interface/Module/Module.h"

#include <memory>

class Scheduler;
class Window;
class RasterDevice;
class WindowParameters;
class SchedulerModule;
class DisplayModule;

class RasterModule : public Module<RasterModule> {

public:

	RasterModule() = default;
	virtual ~RasterModule() = default;

	RasterModule(const RasterModule &) = delete;
	RasterModule(RasterModule &&) noexcept = default;

	RasterModule& operator=(const RasterModule &) = delete;
	RasterModule& operator=(RasterModule &&) noexcept = default;

	virtual void Draw() = 0;
	virtual void DrawIndirect() = 0;

	//Factory
	static std::shared_ptr<RasterModule> CreateModule(std::shared_ptr<SchedulerModule>&,
		std::shared_ptr<DisplayModule>&);


};
