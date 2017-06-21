#pragma once

#include "Event.h"
#include "Multithreading/Scheduler.h"
#include "Metrics.h"

#include <functional>
#include <unordered_map>
#include <string>
#include <memory>
#include <utility>

/* . */
/* . */
namespace EventManager {

/* . */
/* . */
	namespace detail {

		/* Defines an alias representing the event pointer. */
		/* Defines an alias representing the event pointer. */
		typedef std::shared_ptr<BaseEvent> EventPtr;
		/* Defines an alias representing the map. */
		/* Defines an alias representing the map. */
		typedef std::unordered_map<std::string, EventPtr> EMap;

		/* The event map */
		/* The event map */
		extern EMap eventMap;
		/* The identifier */
		/* The identifier */
		extern uint id;

		/*
		 *    Joins.
		 *
		 *    @param	parameter1	The first parameter.
		 *    @param	parameter2	The second parameter.
		 *
		 *    @return	A std::string.
		 */

		std::string Join(std::string, std::string);

	}

	//Listener functions
	template <typename T,
		typename ... Args>

		/*
		 *    Listens.
		 *
		 *    @param 		 	channel 	The channel.
		 *    @param 		 	name		The name.
		 *    @param [in,out]	instance	If non-null, the instance.
		 *    @param [in,out]	func		If non-null, the function.
		 *
		 *    @return	An uint.
		 */

		uint Listen(std::string channel, std::string name, T *instance, void(T::*func)(Args...) const) {
		return Listen(channel, name, [=](Args... args) {
			(instance->*func)(args...);
		});
	}

	template <typename T,
		typename ... Args>

		/*
		 *    Listens.
		 *
		 *    @param 		 	channel 	The channel.
		 *    @param 		 	name		The name.
		 *    @param [in,out]	instance	If non-null, the instance.
		 *    @param [in,out]	func		If non-null, the function.
		 *
		 *    @return	An uint.
		 */

		uint Listen(std::string channel, std::string name, T *instance, void(T::*func)(Args...)) {
		return Listen(channel, name, [=](Args... args) {
			(instance->*func)(args...);
		});
	}

	template <typename Fn, typename... Args >

	/*
	 *    Listens.
	 *
	 *    @param 		 	channel	The channel.
	 *    @param 		 	name   	The name.
	 *    @param [in,out]	func   	The function.
	 *    @param 		 	args   	Variable arguments providing [in,out] The arguments.
	 *
	 *    @return	An uint.
	 */

	uint Listen(std::string channel, std::string name, Fn && func, Args && ... args)
	{

		typedef Event<Args...> EventType;

		std::pair<detail::EMap::iterator, bool> ret;
		ret = detail::eventMap.insert(std::make_pair(detail::Join(channel, name), detail::EventPtr(new EventType())));

		EventType* evt = dynamic_cast<EventType*>(ret.first->second.get());

		if (evt)
		{
			(*evt).Listen(detail::id, func);
		}

		return detail::id++;

	}

	/*
	 *    Removes this object.
	 *
	 *    @param	channel	The channel.
	 *    @param	name   	The name.
	 *    @param	ID	   	The identifier.
	 */

	void Remove(std::string channel, std::string name, int ID);

	/*
	 *    Removes this object.
	 *
	 *    @param	channel	The channel.
	 *    @param	name   	The name.
	 */

	void Remove(std::string channel, std::string name);

	/*
	 *    Emits.
	 *
	 *    @tparam	Args	Type of the arguments.
	 *    @param	channel	The channel.
	 *    @param	name   	The name.
	 *    @param	args   	Variable arguments providing the arguments.
	 *
	 *    ### tparam	Args	Type of the arguments.
	 */

	template <typename... Args>
	void Emit(std::string channel, std::string name, Args... args)
	{
		typedef Event<Args...> EventType;
		detail::EMap::const_iterator itr = detail::eventMap.find(detail::Join(channel, name));
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