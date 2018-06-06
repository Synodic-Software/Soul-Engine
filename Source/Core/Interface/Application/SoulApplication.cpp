#include "SoulApplication.h"
#include "Core/Soul.h"

SoulApplication::SoulApplication(SoulParameters params) :
	hasControl(true),
	parameters(params),
	soul(parameters)
{
}


void SoulApplication::Initialize() {
	
}


void SoulApplication::Terminate() {

}


void SoulApplication::Run() {

	CheckParameters();

	soul.Initialize();

	//EventManager::Listen("Input", "ESCAPE", [](keyState state) {
	//	if (state == RELEASE) {
	//		SoulSignalClose();
	//	}
	//});

	//uint xSize;
	//Settings::Get("MainWindow.Width", uint(800), xSize);
	//uint ySize;
	//Settings::Get("MainWindow.Height", uint(450), ySize);
	//uint xPos;
	//Settings::Get("MainWindow.X_Position", uint(0), xPos);
	//uint yPos;
	//Settings::Get("MainWindow.Y_Position", uint(0), yPos);
	//int monitor;
	//Settings::Get("MainWindow.Monitor", 0, monitor);

	//WindowType type;
	//int typeCast;
	//Settings::Get("MainWindow.Type", static_cast<int>(WINDOWED), typeCast);
	//type = static_cast<WindowType>(typeCast);

	//AbstractWindow* mainWindow = ManagerInterface::Instance().CreateWindow(type, "main", monitor, xPos, yPos, xSize, ySize);

	//ManagerInterface::Instance().SetWindowLayout(mainWindow, new SingleLayout(new RenderWidget()));

	//SoulRun();

	soul.Terminate();
}

void SoulApplication::CheckParameters() {

	//TODO
	
}