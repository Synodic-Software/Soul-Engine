#include "Layout.h"

Layout::Layout()
{
}

Layout::Layout(GLFWwindow* winIn)
{
	UpdateWindow(winIn);
}

Layout::~Layout()
{
}

void Layout::Draw(GLFWwindow* window) {
	for (auto& wid : widgets) {
		wid->Draw(window);
	}
}

void  Layout::UpdateWindow(GLFWwindow* winIn) {
	window = winIn;

	for (auto& wid : widgets) {
		wid->UpdateWindow(winIn);
	}
}
