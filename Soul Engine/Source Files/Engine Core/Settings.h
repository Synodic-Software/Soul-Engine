#pragma once

#include <string>
#include <map>

class Settings{
public:
	Settings(std::string);
	int  Retrieve(std::string setting);
	int  Retrieve(std::string setting, int defaultSet);
	void Set(std::string setting, int value);

private:
	void Update();
	std::map<std::string, int> values;
	std::string fileName;
};