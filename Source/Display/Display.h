#pragma once

#include "Core/Interface/Module/Module.h"
#include "WindowParameters.h"

#include <memory>

class Window;
class RasterBackend;

class Display : public Module {

public:

	Display();
	virtual ~Display() = default;

	Display(const Display&) = delete;
	Display(Display&&) noexcept = default;

	Display& operator=(const Display&) = delete;
	Display& operator=(Display&&) noexcept = default;


	virtual void Draw() = 0;
	virtual bool Active() = 0;

	virtual void CreateWindow(const WindowParameters&, RasterBackend*) = 0;
	virtual void RegisterRasterBackend(RasterBackend*) = 0;


	//Factory
	static std::unique_ptr<Display> CreateModule();


protected:

	bool active_;

};
