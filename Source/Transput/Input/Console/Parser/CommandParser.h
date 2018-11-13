#include <map>
#include <string>
#include <list>
#include <iostream>

class CommandParser {
public:
	static std::map<std::string,std::string> parse(std::istream& istr_);
private:
	static bool parse_property(std::map<std::string,std::string>& properties, std::list<std::string>::const_iterator& itr,
			const std::list<std::string>::const_iterator command_end);
};
