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
	RasterBackend::Draw(windowHandle, widgetJob);
}

void Widget::UpdateWindow(GLFWwindow* winIn) {
	window = winIn;
}
