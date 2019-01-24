#include "CLIConsoleManager.h"
#include "Display/Window.h"

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

	std::map<std::string,std::string> properties = CommandParser::parse(istr_);

	if (properties["err"] != "") {
		estr_ << "ERROR: " << properties["err"] << std::endl;
		return false;
	}

	if (command == "load_window") {
		// Assign default stats for the window
		int w_width=512, w_height=512;
		std::string w_name = "Main";

		if (properties["w"] != "") w_width = stoi(properties["w"]);
		else if (properties["width"] != "") w_width = stoi(properties["width"]);

		if (properties["h"] != "") w_height = stoi(properties["h"]);
		else if (properties["height"] != "") w_height = stoi(properties["height"]);

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

		ostr_ << "Creating window of width " << w_width << " and height " << w_height
			<< " named '" << w_name << "'" << std::endl;

		soul.CreateWindow(windowParams);
		soul.Run();
	} else {
		estr_ << "ERROR: \"" << command << "\" is not a valid command" << std::endl;
		return false;
	}

	return true;

}
