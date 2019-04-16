#pragma once

#include "Transput/Input/Desktop/DesktopInputManager.h"

#include "Composition/Event/EventManager.h"
#include "Composition/Entity/EntityManager.h"
#include "Frame/FramePipeline.h"

#include <variant>

class Soul;
class InputManager;
class WindowManager;
class ConsoleManager;

//TODO: remove once c++20 modules are integrated
class Soul::Implementation
{

public:

	//monostate allows for empty construction
	using inputManagerVariantType = std::variant<std::monostate, DesktopInputManager>;
	using consoleManagerVariantType = std::variant<std::monostate, CLIConsoleManager>;

	Implementation(Soul&); 
	~Implementation();

	//TODO: Old managers should be transferred to `modules` (different organizational style)
	EntityManager entityManager_;
	EventManager eventManager_;
	inputManagerVariantType inputManagerVariant_;
	InputManager* inputManager_;

	FramePipeline<3> framePipeline_;

private:

	inputManagerVariantType ConstructInputManager();
	InputManager* ConstructInputPtr();

};
