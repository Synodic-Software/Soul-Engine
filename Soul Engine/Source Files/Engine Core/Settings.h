#pragma once

#include <string>
#include <map>

class Settings{
public:
	Settings(std::string);
	std::string  Retrieve(std::string);
	void Set(std::string setting, std::string value);

private:
	void Update();
	std::map<std::string, std::string> values;
	std::string fileName;
};