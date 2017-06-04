#pragma once

#include "Event.h"
#include "Multithreading/Scheduler.h"
#include "Metrics.h"

#include <functional>
#include <map>
#include <string>
#include <memory>
#include <utility>

namespace EventManager {

	namespace detail {

		typedef std::shared_ptr<BaseEvent> EventPtr;
		typedef std::map<std::string, EventPtr> EMap;

		extern EMap eventMap;
		extern uint id ;

	}

	////Listener functions
	//template <typename Fn,
	//	typename ... Args>
	//	void Listen(int ID, T *instance, void(T::*func)(Args...) const) {
	//	return Listen(ID, [=](Args... args) {
	//		(instance->*func)(args...);
	//	});
	//}

	//template <typename Fn,
	//	typename ... Args>
	//	void Listen(int ID, T *instance, void(T::*func)(Args...)) {
	//	return Listen(ID, [=](Args... args) {
	//		(instance->*func)(args...);
	//	});
	//}

	//template <typename ... Args>
	//void Listen(int ID, const std::function<void(Args...)>& fn) {
	//	listeners.insert(std::make_pair(ID, fn));
	//}

	//template <typename ... Args>
	//void Listen(int ID, std::function<void(Args...)>&& fn) {
	//	listeners.insert(std::make_pair(ID, fn));
	//}

	template <typename T, typename... Args >
	uint Listen(std::string name,T *instance, void(T::*func)(Args...))
	{

		typedef Event<std::forward<Args>(func)...> EventType;

		std::pair<detail::EMap::iterator, bool> ret;
		ret = detail::eventMap.insert(std::make_pair(name, detail::EventPtr(new EventType())));

		EventType* evt = dynamic_cast<EventType*>(ret.first->second.get());

		if (evt)
		{
			(*evt).Listen(detail::id, instance, func);
		}

		return detail::id++;

	}

	void Remove(std::string name, int ID);

	void Remove(std::string name);

	template <typename... Args>
	void Emit(std::string name, Args... args)
	{
		typedef Event<Args...> EventType;
		detail::EMap::const_iterator itr = detail::eventMap.find(name);
		if (itr != detail::eventMap.end())
		{
			EventType* evt = dynamic_cast<EventType*>(itr->second.get());
			if (evt)
			{
				(*evt).Emit(args...);
			}
		}
	}


}