#pragma once

#include "Core/Interface/Module/Module.h"

#include <memory>

class Window;
class RasterDevice;
class WindowParameters;
class Display;

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
	virtual void CreateWindow(const WindowParameters&) = 0;

	//Factory
	static std::unique_ptr<RasterBackend> CreateModule();


protected:

	static std::unique_ptr<Display> displayModule_;


};
