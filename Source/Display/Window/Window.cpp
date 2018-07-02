#include "Window.h"

Window::Window(WindowParameters& params) :
	context_(nullptr),
	layout_(nullptr),
	windowParams_(params)
{
}

std::any& Window::GetContext() {
	return context_;
}
