#include "..\..\Source Files\Utility\Logger.h"
#include <iostream>
#include <cassert>

int main()
{
	std::string line;
	Logger::WriteFile();
	std::ifstream file("Engine.log");
	std::getline(file, line);
	assert(line.compare("File: Engine.log Line: 7 | ") == 0);
	file.close();
	return 0;
}