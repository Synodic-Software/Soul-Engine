#include "EventManager.h"

namespace EventManager {

	namespace detail {

		typedef std::shared_ptr<BaseEvent> EventPtr;
		typedef std::map<std::string, EventPtr> EMap;

		EMap eventMap;
		uint id = 0;

	}
}