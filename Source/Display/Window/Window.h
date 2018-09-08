#pragma once

#include "Display/Layout/Layout.h"
#include "Rasterer/Graphics API/SwapChain.h"
#include "Composition/Entity/Entity.h"

#include <string>
#include <any>

enum class WindowType { WINDOWED, FULLSCREEN, BORDERLESS, EMPTY };

class Surface;

struct WindowParameters {

	WindowType type;
	std::string title;
	glm::uvec2 pixelPosition;
	glm::uvec2 pixelSize;
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
	virtual void FrameBufferResize(int, int) = 0;
	virtual void SetLayout(Layout*) = 0;

	std::any& GetContext();

protected:

	//todo abstract instead of std::any
	std::any context_;

	std::unique_ptr<Layout> layout_;
	Entity swapChain_;
	Entity surface_;
	Entity device_;

	WindowParameters windowParams_;

};