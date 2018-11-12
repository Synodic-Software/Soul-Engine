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

	std::list<std::string> input;
	std::string tmp;
	while (istr_ >> tmp) input.push_back(tmp);

	std::map<std::string,std::string> properties = CommandParser.parse(input);

	if (command == "load_window") {
		// Assign default stats for the window
		int w_width=512, w_height=512;
		std::string w_name = "Main";

		if (properties["w"] != "") w_width = atoi(properties["w"]);
		else if (properties["width"] != "") w_width = atoi(properties["width"]);

		if (properties["h"] != "") w_height = atoi(properties["h"]);
		else if (properties["height"] != "") w_height = atoi(properties["height"]);

		if (properties["name"] != "") w_name = properties["name"];

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
		estr_ << "\"" << command << "\" is not a valid command!" << std::endl;
		return false;
	}

	return true;
}
