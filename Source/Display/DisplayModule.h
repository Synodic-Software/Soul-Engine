#pragma once

#include "Core/Interface/Module/Module.h"
#include "WindowParameters.h"
#include "GUI/GUIModule.h"

#include <memory>

class Window;
class RasterModule;

class DisplayModule : public Module<DisplayModule> {

public:

	DisplayModule();
	virtual ~DisplayModule() = default;

	DisplayModule(const DisplayModule&) = delete;
	DisplayModule(DisplayModule&&) noexcept = default;

	DisplayModule& operator=(const DisplayModule&) = delete;
	DisplayModule& operator=(DisplayModule&&) noexcept = default;


	virtual void Draw() = 0;
	virtual bool Active() = 0;

	virtual void CreateWindow(const WindowParameters&, RasterModule*) = 0;
	virtual void RegisterRasterBackend(RasterModule*) = 0;


	//Factory
	static std::unique_ptr<DisplayModule> CreateModule();


protected:

	bool active_;

	std::unique_ptr<GUIModule> gui_;

};
