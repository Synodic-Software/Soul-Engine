#include "SoulInfo.h"

SoulInfo::SoulInfo(SoulParameters params) :
	SoulApplication(params) {

}

int main(int, char*[])
{
	//app params
	SoulParameters appParams;
	SoulInfo app(appParams);

	//create the window
	WindowParameters windowParams;
	windowParams.type = WindowType::WINDOWED;
	windowParams.title = "Main";
	windowParams.monitor = 0;
	windowParams.pixelPosition.x = 0;
	windowParams.pixelPosition.y = 0;
	windowParams.pixelSize.x = 512;
	windowParams.pixelSize.y = 512;

	app.CreateWindow(windowParams);

	app.Run();

}