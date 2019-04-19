#pragma once

#include "Core/Interface/Module/Module.h"
#include "WindowParameters.h"
#include "Display/GUI/GUIModule.h"

#include <memory>

class Window;
class RasterModule;

class WindowModule : public Module<WindowModule> {

public:

	WindowModule();
	virtual ~WindowModule() = default;

	WindowModule(const WindowModule&) = delete;
	WindowModule(WindowModule&&) noexcept = default;

	WindowModule& operator=(const WindowModule&) = delete;
	WindowModule& operator=(WindowModule&&) noexcept = default;


	virtual void Draw() = 0;
	virtual bool Active() = 0;

	virtual void CreateWindow(const WindowParameters&, RasterModule*) = 0;
	virtual void RegisterRasterBackend(RasterModule*) = 0;


	//Factory
	static std::unique_ptr<WindowModule> CreateModule();


protected:

	bool active_;

	std::unique_ptr<GUIModule> gui_;

};
