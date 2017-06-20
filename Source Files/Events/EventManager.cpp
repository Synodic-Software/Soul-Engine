//---------------------------------------------------------------------------------------------------
//@file	N:\Documents\Soul Engine\Source Files\Events\EventManager.cpp.
//Implements the event manager class.

#include "EventManager.h"

namespace EventManager {

	namespace detail {

		//Defines an alias representing the event pointer.
		typedef std::shared_ptr<BaseEvent> EventPtr;
		//Defines an alias representing the map.
		typedef std::unordered_map<std::string, EventPtr> EMap;

		//The event map
		EMap eventMap;
		//The identifier
		uint id = 0;

		//---------------------------------------------------------------------------------------------------
		//Joins.
		//@param	a	A std::string to process.
		//@param	b	A std::string to process.
		//@return	A std::string.

		std::string Join(std::string a, std::string b) {
			return a + ":" + b;
		}
	}

	//---------------------------------------------------------------------------------------------------
	//Removes this object.
	//@param	channel	The channel.
	//@param	name   	The name.
	//@param	ID	   	The identifier.

	void Remove(std::string channel, std::string name, int ID) {
		detail::EMap::const_iterator itr = detail::eventMap.find(detail::Join(channel, name));
		if (itr != detail::eventMap.end())
		{
			itr->second.get()->Remove(ID);
		}
	}

	//---------------------------------------------------------------------------------------------------
	//Removes this object.
	//@param	channel	The channel.
	//@param	name   	The name.

	void Remove(std::string channel, std::string name) {
		detail::eventMap.erase(detail::Join(channel, name));
	}
}