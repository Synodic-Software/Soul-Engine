#pragma once

#include "Core/Interface/Module/Module.h"

#include "Types.h"

#include <memory>
#include <any>

class Scheduler;
class Window;
class RasterDevice;
class WindowParameters;
class SchedulerModule;
class WindowModule;

class RasterModule : public Module<RasterModule> {

public:

	RasterModule() = default;
	virtual ~RasterModule() = default;

	RasterModule(const RasterModule &) = delete;
	RasterModule(RasterModule &&) noexcept = default;

	RasterModule& operator=(const RasterModule &) = delete;
	RasterModule& operator=(RasterModule &&) noexcept = default;


	virtual void Render() = 0;

	virtual uint RegisterSurface(std::any, glm::uvec2) = 0;
	virtual void UpdateSurface(uint, glm::uvec2) = 0;
	virtual void RemoveSurface(uint) = 0;

	//Agnostic raster API interface
	virtual void Draw() = 0;
	virtual void DrawIndirect() = 0;
	virtual void UpdateBuffer() = 0;
	virtual void UpdateTexture() = 0;
	virtual void CopyBuffer() = 0;
	virtual void CopyTexture() = 0;


	//Factory
	static std::shared_ptr<RasterModule> CreateModule(std::shared_ptr<SchedulerModule>&,
		std::shared_ptr<WindowModule>&);


};
