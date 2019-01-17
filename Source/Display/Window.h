#pragma once

#include "WindowParameters.h"
#include "Composition/Entity/Entity.h"


class Window
{

public:

	Window(const WindowParameters&);
	virtual ~Window() = default;

	Window(const Window &) = delete;
	Window(Window &&) noexcept = default;

	Window& operator=(const Window &) = delete;
	Window& operator=(Window &&) noexcept = default;

	WindowParameters& Parameters();

protected:

	Entity layout_;
	WindowParameters windowParams_;

};
