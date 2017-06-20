#include "EventManager.h"

namespace EventManager {

	namespace detail {

		typedef std::shared_ptr<BaseEvent> EventPtr;
		typedef std::unordered_map<std::string, EventPtr> EMap;

		EMap eventMap;
		uint id = 0;

		std::string Join(std::string a, std::string b) {
			return a + ":" + b;
		}
	}

	void Remove(std::string channel, std::string name, int ID) {
		detail::EMap::const_iterator itr = detail::eventMap.find(detail::Join(channel, name));
		if (itr != detail::eventMap.end())
		{
			itr->second.get()->Remove(ID);
		}
	}

	void Remove(std::string channel, std::string name) {
		detail::eventMap.erase(detail::Join(channel, name));
	}
}