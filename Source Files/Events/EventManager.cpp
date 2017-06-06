#include "EventManager.h"

namespace EventManager {

	namespace detail {

		typedef std::shared_ptr<BaseEvent> EventPtr;
		typedef std::unordered_map<std::string, EventPtr> EMap;

		EMap eventMap;
		uint id = 0;

	}

	void Remove(std::string name, int ID) {
		detail::EMap::const_iterator itr = detail::eventMap.find(name);
		if (itr != detail::eventMap.end())
		{
			itr->second.get()->Remove(ID);
		}
	}

	void Remove(std::string name) {
		detail::eventMap.erase(name);
	}
}