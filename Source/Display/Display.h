#pragma once

#include "Core/Interface/Module/Module.h"

#include <memory>
#include <variant>

class WindowParameters;
class Window;

class Display : public Module {

public:

	virtual ~Display() = default;

	virtual void Draw() = 0;
	virtual std::shared_ptr<Window> CreateWindow(WindowParameters&) = 0;
	virtual bool ShouldClose() = 0;

	//Factory
	static std::shared_ptr<Display> CreateModule();

};
