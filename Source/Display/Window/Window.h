#pragma once

#include "Core/Utility/Types.h"
#include <Display/Layout/Layout.h>

#include <string>
#include <memory>
#include <any>

enum class WindowType { WINDOWED, FULLSCREEN, BORDERLESS, EMPTY };

struct WindowParameters {

	WindowType type;
	std::string title;
	uint pixelPosX;
	uint pixelPosY;
	uint pixelWidth;
	uint pixelHeight;
	int monitor;

};

class Window
{

public:

	Window(WindowParameters&);
	virtual ~Window() = default;

	Window(const Window &) = delete;
	Window(Window &&) noexcept = default;

	Window& operator=(const Window &) = delete;
	Window& operator=(Window &&) noexcept = default;

	virtual void Draw() = 0;

	virtual void Refresh() = 0;
	virtual void Close() = 0;
	virtual void Resize(int, int) = 0;
	virtual void PositionUpdate(int, int) = 0;

	virtual void SetLayout(Layout*) = 0;

	std::any& GetContext();

protected:

	std::any context_;
	std::unique_ptr<Layout> layout_;

	WindowParameters windowParams_;

};