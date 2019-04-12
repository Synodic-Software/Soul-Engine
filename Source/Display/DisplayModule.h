#pragma once

#include "Core/Interface/Module/Module.h"
#include "WindowParameters.h"

#include <memory>

class Window;
class RasterBackend;
class GUIModule;

class DisplayModule : public Module<DisplayModule> {

public:

	DisplayModule();
	virtual ~DisplayModule();

	DisplayModule(const DisplayModule&) = delete;
	DisplayModule(DisplayModule&&) noexcept = default;

	DisplayModule& operator=(const DisplayModule&) = delete;
	DisplayModule& operator=(DisplayModule&&) noexcept = default;


	virtual void Draw() = 0;
	virtual bool Active() = 0;

	virtual void CreateWindow(const WindowParameters&, RasterBackend*) = 0;
	virtual void RegisterRasterBackend(RasterBackend*) = 0;


	//Factory
	static std::unique_ptr<DisplayModule> CreateModule();


protected:

	bool active_;

	std::unique_ptr<GUIModule> gui_;

};
