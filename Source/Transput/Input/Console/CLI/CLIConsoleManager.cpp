#include "CLIConsoleManager.h"
#include "Display/Window/Window.h"

CLIConsoleManager::CLIConsoleManager(EventManager& eventManager, Soul& soul_ref) :
	ConsoleManager(eventManager, soul_ref),
	istr_(std::cin),
	ostr_(std::cout),
	estr_(std::cerr)
{
}

void CLIConsoleManager::Poll() {

	std::string command;
	while (istr_ >> command) {
		ProcessCommand(command);
	}

}

bool CLIConsoleManager::ProcessCommand(const std::string& command) {

	if (command == "load_window") {
		// Read in stats about the window
		int w_width, w_height;
		std::string w_name;
		istr_ >> w_width >> w_height >> w_name;

		// Create and display the window
		WindowParameters windowParams;
		windowParams.type = WindowType::WINDOWED;
		windowParams.title = w_name;
		windowParams.monitor = 0;
		windowParams.pixelPosition.x = 0;
		windowParams.pixelPosition.y = 0;
		windowParams.pixelSize.x = w_width;
		windowParams.pixelSize.y = w_height;

		soul.CreateWindow(windowParams);
		soul.Run();
	} else {
		estr_ << "\""<< command << "\" is not a valid command!" << std::endl;
		return false;
	}

	// Clear any excess inputs to istr_
	std::string tmp;
	while (istr_ >> tmp);

	return true;
}
