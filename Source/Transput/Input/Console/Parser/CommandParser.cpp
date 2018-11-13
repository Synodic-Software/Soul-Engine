#include "CommandParser.h"

static std::map<std::string, std::string> CommandParser::parse(std::istream& istr_) {

	std::map<std::string,std::string> properties;
	std::list<std::string> parsed_command;
	
	char bracket = '';
	std::string tmp;
	while (istr_ >> tmp) {
		if (bracket != '') {
			if (tmp.back() == bracket) {
				tmp = tmp.substr(0,tmp.size()-1);
				bracket = '';
			}
			parsed_command.back() += tmp;
		} else {
			char first_char = tmp.front();
			if (first_char=='\'' || first_char=='"') {
				bracket = first_char;
				parse_command.push_back(tmp.substr(1));
			} else {
				parse_command.push_back(tmp);
			}
		}
	}

	// Ensure we have proper syntax
	if (bracket != '') {
		properties["err"] = "missing ending " + bracket;
		return properties;
	}

	std::list<std::string>::const_iterator itr = parse_command.cbegin();
	while (itr != parsed_command.cend()) {
		if (!parse_property(command_properties, itr, parse_command.cend())) {
			// An error was detected
			return properties;
		}
		itr++;
	}

	return properties;

}

static bool CommandParser::parse_property(std::map<std::string, std::string>& properties, std::list<std::string>::const_iterator& itr,
		const std::list<std::string>::const_iterator& command_end) {

	const std::string& property = *itr;
	if (property[1] != '-') {
		// This is a property given in the form '-property VAL'
		itr++;
		if (itr == command_end) {
			// That was the the last command
			properties["err"] = "property '" + property "' is missing a value";
		}
		properties[property.substr(1)] = *itr;
	} else {
		// This is a property given in the form '--property=VAL'
		size_t val_index = property.find('=');
		if (val_index == std::string::npos) {
			// '=' was not found
			properties["err"] = "invalid property '" + property + "'";
			return false;
		}
		std::string val = property.substr(val_index+1);
		properties[property.substr(2,val_index)] = val;
	}

	return true;

}
