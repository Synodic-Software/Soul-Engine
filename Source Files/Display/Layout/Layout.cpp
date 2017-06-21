#include "Layout.h"

Layout::Layout()
{
}

Layout::~Layout()
{
}

void Layout::Draw() {
	for (auto& wid : widgets) {
		wid->Draw();
	}
}

void Layout::UpdateWindow(GLFWwindow* winIn) {
	window = winIn;

	for (auto& wid : widgets) {
		wid->UpdateWindow(winIn);
	}
}

void Layout::UpdatePositioning( glm::uvec2 newPosition, glm::uvec2 newSize) {
	size = newSize;
	position = newPosition;
	for (auto& wid : widgets) {
		wid->UpdatePositioning(newPosition, newSize);
	}

}

void Layout::RecreateData() {
	for (auto& wid : widgets) {
		wid->RecreateData();
	}
}
