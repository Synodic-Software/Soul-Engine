#include "Widget.h"
#include "Raster Engine\RasterBackend.h"

void Widget::Draw()
{

}

void Widget::UpdateWindow(GLFWwindow* winIn) {
	window = winIn;
}

void Widget::UpdatePositioning( glm::uvec2 newPosition, glm::uvec2 newSize) {
	size = newSize;
	position = newPosition;
}

void Widget::RecreateData() {

}
