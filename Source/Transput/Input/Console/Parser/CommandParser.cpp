#include "CommandParser.h"

static std::map<std::string, std::string> CommandParser::parse(const std::list<std::string>& command) {

	std::map<std::string,std::string> properties;
	std::list<std::string>::const_iterator itr = command.cbegin();

	while (itr != command.cend()) {
		parse_property(command_properties, itr);
		itr++;
	}

	return properties;

}

static void CommandParser::parse_property(std::map<std::string, std::string>& properties, std::list<std::string>::const_iterator& itr) {

	// TODO: Add proper error handling
	// TODO: Add support for multi-word values
	const std::string& property = *itr;
	if (property[1] != '-') {
		// This is a property given in the form '-property VAL'
		itr++;
		properties[property.substr(1)] = *itr;
	} else {
		// This is a property given in the form '--property=VAL'
		size_t val_index = property.find('='); // TODO: Support std::string::npos ('=' not found)
		std::string val = property.substr(val_index+1);
		properties[property.substr(2,val_index)] = val;
	}

}
