//---------------------------------------------------------------------------------------------------
//@file	N:\Documents\Soul Engine\Source Files\Input\InputManager.h.
//Declares the input manager class.

#pragma once

#include "InputSet.h"

//.
namespace InputManager
{
//.
	namespace detail
	{
		//The set in use
		extern InputSet* setInUse;
	}

	//Listener functions
	template <typename T,
	          typename ... Args>

	//---------------------------------------------------------------------------------------------------
	//Listens.
	//@param 		 	channel 	The channel.
	//@param 		 	name		The name.
	//@param [in,out]	instance	If non-null, the instance.
	//@param [in,out]	func		If non-null, the function.
	//@return	An uint.

	uint Listen(std::string channel, std::string name, T* instance, void (T::*func)(Args ...) const)
	{
		return Listen(channel, name, [=](Args ... args)
	              {
		              (instance ->* func)(args...);
	              });
	}

	template <typename T,
	          typename ... Args>

	//---------------------------------------------------------------------------------------------------
	//Listens.
	//@param 		 	channel 	The channel.
	//@param 		 	name		The name.
	//@param [in,out]	instance	If non-null, the instance.
	//@param [in,out]	func		If non-null, the function.
	//@return	An uint.

	uint Listen(std::string channel, std::string name, T* instance, void (T::*func)(Args ...))
	{
		return Listen(channel, name, [=](Args ... args)
	              {
		              (instance ->* func)(args...);
	              });
	}

	template <typename Fn, typename... Args>

	//---------------------------------------------------------------------------------------------------
	//Listens.
	//@param 		 	channel	The channel.
	//@param 		 	name   	The name.
	//@param [in,out]	func   	The function.
	//@param 		 	args   	Variable arguments providing [in,out] The arguments.
	//@return	An uint.

	uint Listen(std::string channel, std::string name, Fn&& func, Args&& ... args)
	{
		typedef Event<Args...> EventType;

		std::pair<detail::EventMap::iterator, bool> ret =
			detail::eventMap.insert(std::make_pair(detail::Join(channel, name), detail::EventPtr(new EventType())));

		EventType* evt = dynamic_cast<EventType*>(ret.first->second.get());

		if (evt)
		{
			(*evt).Listen(detail::id, func);
		}

		return detail::id++;
	}

	//---------------------------------------------------------------------------------------------------
	//Removes this object.
	//@param	channel	The channel.
	//@param	key	   	The key.
	//@param	ID	   	The identifier.

	void Remove(std::string channel, std::string key, int ID);

	//---------------------------------------------------------------------------------------------------
	//Removes this object.
	//@param	channel	The channel.
	//@param	key	   	The key.

	void Remove(std::string channel, std::string key);

	template <typename... Args>

	//---------------------------------------------------------------------------------------------------
	//Emits.
	//@param	channel	The channel.
	//@param	name   	The name.
	//@param	args   	Variable arguments providing the arguments.

	void Emit(std::string channel, std::string name, Args ... args)
	{
		typedef Event<Args...> EventType;
		detail::EventMap::const_iterator itr = detail::eventMap.find(detail::Join(channel, name));
		if (itr != detail::eventMap.end())
		{
			EventType* evt = dynamic_cast<EventType*>(itr->second.get());
			if (evt)
			{
				(*evt).Emit(args...);
			}
		}
	}

	//---------------------------------------------------------------------------------------------------
	//Attach input set.
	//@param [in,out]	newSet	If non-null, set the new belongs to.

	void AttachInputSet(InputSet* newSet)
	{
		detail::setInUse = newSet;
	}
};
