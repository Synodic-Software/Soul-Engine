#include <map>
#include <string>
#include <list>

class CommandParser {
public:
	static std::map<std::string,std::string> parse(const std::string& command);
private:
	static load_window(const std::list<std::string>& command);
};
