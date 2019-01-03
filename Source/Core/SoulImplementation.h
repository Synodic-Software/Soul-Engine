#pragma once

#include "Display/Window/Desktop/DesktopWindowManager.h"

#include "Transput/Input/Desktop/DesktopInputManager.h"
#include "Transput/Input/Console/CLI/CLIConsoleManager.h"

#include "Composition/Event/EventManager.h"
#include "Parallelism/Fiber/Scheduler.h"
#include "Composition/Entity/EntityManager.h"
#include "Rasterer/RasterManager.h"
#include "Frame/FramePipeline.h"

#include <variant>

class Soul;
class InputManager;
class WindowManager;
class ConsoleManager;

class DisplayWindowAPI;
class RasterBackendAPI;

//TODO: remove once c++20 modules are integrated
class Soul::Implementation
{

public:

	//monostate allows for empty construction
	using inputManagerVariantType = std::variant<std::monostate, DesktopInputManager>;
	using windowManagerVariantType = std::variant<std::monostate, DesktopWindowManager>;
	using consoleManagerVariantType = std::variant<std::monostate, CLIConsoleManager>;

	Implementation(Soul&); 
	~Implementation();

	//services and modules	
	std::shared_ptr<DisplayWindowAPI> displayModule;
	std::shared_ptr<RasterBackendAPI> rasterModule;

	//TODO: Old managers should be transferred to `modules` (different organizational style)
	EntityManager entityManager_;
	Scheduler scheduler_;
	EventManager eventManager_;
	inputManagerVariantType inputManagerVariant_;
	InputManager* inputManager_;
	windowManagerVariantType windowManagerVariant_;
	WindowManager* windowManager_;
	RasterManager rasterManager_;
	consoleManagerVariantType consoleManagerVariant_;
	ConsoleManager* consoleManager_;

	FramePipeline<3> framePipeline_;

private:

	inputManagerVariantType ConstructInputManager();
	InputManager* ConstructInputPtr();

	windowManagerVariantType ConstructWindowManager();
	WindowManager* ConstructWindowPtr();

	consoleManagerVariantType ConstructConsoleManager(Soul&);
	ConsoleManager* ConstructConsolePtr();
};
