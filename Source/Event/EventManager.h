#pragma once

#include "Event.h"
#include "Parallelism/Scheduler.h"
#include "Metrics.h"

#include <functional>
#include <unordered_map>
#include <string>
#include <memory>
#include <utility>
#include <type_traits>

/* . */
namespace EventManager {

	/* . */
	namespace detail {

		/* Defines an alias representing the event pointer. */
		typedef std::shared_ptr<BaseEvent> EventPtr;
		/* Defines an alias representing the map. */
		typedef std::unordered_map<std::string, EventPtr> EMap;

		/* The event map */
		extern EMap eventMap;
		/* The identifier */
		extern uint id;

		/*
		 *    Joins.
		 *    @param	parameter1	The first parameter.
		 *    @param	parameter2	The second parameter.
		 *    @return	A std::string.
		 */

		std::string Join(std::string, std::string);

		/*
		*    An identity.
		*    @tparam	T	Generic type parameter.
		*/

		template<typename T>
		struct identity
		{
			using type = void;
		};

		/*
		*    A *)( args...) const>.
		*    @tparam	Ret  	Type of the ret.
		*    @tparam	Class	Type of the class.
		*    @tparam	Args 	Type of the arguments.
		*/

		template<typename Ret, typename Class, typename... Args>
		struct identity<Ret(Class::*)(Args...) const>
		{
			using type = std::function<Ret(Args...)>;
		};

		/*
		*    A *)( args...)>.
		*    @tparam	Ret  	Type of the ret.
		*    @tparam	Class	Type of the class.
		*    @tparam	Args 	Type of the arguments.
		*/

		template<typename Ret, typename Class, typename... Args>
		struct identity<Ret(Class::*)(Args...)>
		{
			using type = std::function<Ret(Args...)>;
		};

		/*
		*    Listens.
		*    @param 		 	channel	The channel.
		*    @param 		 	name   	The name.
		*    @param [in,out]	func   	The function.
		*    @param 		 	args   	Variable arguments providing [in,out] The arguments.
		*    @return	An uint.
		*/

		template <typename... Args>
		uint ListenBase(std::string channel, std::string name, const std::function<void(Args...)>& func)
		{

			typedef Event<Args...> EventType;

			auto ret = detail::eventMap.insert(std::make_pair(detail::Join(channel, name), detail::EventPtr(new EventType())));

			EventType* evt = dynamic_cast<EventType*>(ret.first->second.get());

			if (evt)
			{
				evt->Listen(detail::id, func);
			}

			return detail::id++;

		}

		/*
		*    Listens.
		*    @param 		 	channel	The channel.
		*    @param 		 	name   	The name.
		*    @param [in,out]	func   	The function.
		*    @param 		 	args   	Variable arguments providing [in,out] The arguments.
		*    @return	An uint.
		*/

		template <typename... Args>
		uint ListenBase(std::string channel, std::string name, std::function<void(Args...)>&& func)
		{

			typedef Event<Args...> EventType;

			auto ret = detail::eventMap.insert(std::make_pair(detail::Join(channel, name), detail::EventPtr(new EventType())));

			EventType* evt = dynamic_cast<EventType*>(ret.first->second.get());

			if (evt)
			{
				evt->Listen(detail::id, func);
			}

			return detail::id++;

		}
	}


	/*
	*    Listens.
	*    @param 		 	channel	The channel.
	*    @param 		 	name   	The name.
	*    @param [in,out]	func   	The function.
	*    @param 		 	args   	Variable arguments providing [in,out] The arguments.
	*    @return	An uint.
	*/

	template <typename... Args>
	uint Listen(std::string channel, std::string name, void(*func)(Args...))
	{
		std::function<void(Args...)> f = func;
		return detail::ListenBase(channel, name, f);
	}

	/*
	 *    Listens.
	 *    @param	channel	The channel.
	 *    @param	name   	The name.
	 *    @param	func   	The function.
	 *    @return	A uint. Disables the function template if the function is not a lambda
	 */

	template<typename Fn>
	uint Listen(std::string channel, std::string name, Fn const &func)
	{
		typename detail::identity<decltype(&Fn::operator())>::type
			newFunc = func;

		return detail::ListenBase(channel, name, newFunc);
	}

	/*
	*    Listens.
	*    @param	channel	The channel.
	*    @param	name   	The name.
	*    @param	func   	The function.
	*    @return	A uint. Disables the function template if the function is not a lambda
	*/

	template<typename Fn>
	uint Listen(std::string channel, std::string name, Fn && func)
	{
		typename detail::identity<decltype(&Fn::operator())>::type
			newFunc = func;

		return detail::ListenBase(channel, name, newFunc);
	}

	/*
	 *    Removes this object.
	 *    @param	channel	The channel.
	 *    @param	name   	The name.
	 *    @param	ID	   	The identifier.
	 */

	void Remove(std::string channel, std::string name, int ID);

	/*
	 *    Removes this object.
	 *    @param	channel	The channel.
	 *    @param	name   	The name.
	 */

	void Remove(std::string channel, std::string name);

	/*
	 *    Emits.
	 *    @tparam	Args	Type of the arguments.
	 *    @param	channel	The channel.
	 *    @param	name   	The name.
	 *    @param	args   	Variable arguments providing the arguments.
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