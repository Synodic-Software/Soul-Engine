#include "Widget.h"
#include "Raster Engine\RasterBackend.h"

Widget::Widget()
{
	widgetJob=RasterBackend::CreateJob();
}

Widget::~Widget()
{

}

void Widget::Draw(GLFWwindow* windowHandle)
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
