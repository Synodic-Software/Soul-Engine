#pragma once

#include "Core/Interface/Module/Module.h"
#include "Display/WindowParameters.h"

#include <memory>

class Window;

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

	virtual std::shared_ptr<Window> CreateWindow(WindowParameters&) = 0;


	//Factory
	static std::shared_ptr<Display> CreateModule();


protected:

	bool active_;

};
