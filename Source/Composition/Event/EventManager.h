#pragma once

#include "Event.h"
#include "Parallelism/Fiber/Scheduler.h"
#include "Core/Utility/Types.h"
#include "Core/Utility/HashString/HashString.h"

#include <unordered_map>
#include <memory>

//TODO replace with Callable Concept c++20
#include <boost/callable_traits.hpp>

//signal and slots system
class EventManager {

public:

	EventManager();
	~EventManager() = default;

	template<typename Fn>
	uint64 Listen(HashString::HashType, HashString::HashType, Fn&&);

	template <typename... Args>
	void Emit(HashString::HashType, HashString::HashType, Args&&...);

	void Remove(HashString::HashType, HashString::HashType, uint64);
	void Remove(HashString::HashType, HashString::HashType);
	void Remove(HashString::HashType);


private:

	using EventPtr = std::unique_ptr<BaseEvent>;
	using EventHashMap = std::unordered_map<
		HashString::HashType,
		std::unordered_map<HashString::HashType, EventPtr>
	>;

	//maps a hashedstring (as an int type) to an event
	EventHashMap eventMap_;
	uint64 id_;

};

template<typename Fn>
uint64 EventManager::Listen(HashString::HashType channel, HashString::HashType name, Fn&& func)
{
	using type = boost::callable_traits::function_type_t<Fn>;

	auto& baseEventPtr = eventMap_[channel][name];
	if (!baseEventPtr.get()) {
		baseEventPtr = std::make_unique<Event<type>>();
	}

	auto event = static_cast<Event<type>*>(baseEventPtr.get());

	event->Listen(id_, std::forward<Fn>(func));

	return id_++;
}

template <typename... Args>
void EventManager::Emit(HashString::HashType channel, HashString::HashType name, Args&&... args)
{

	//TODO cast return type
	auto event = static_cast<Event<void(decltype(args)...)>*>(eventMap_[channel][name].get());
	if (event)
	{
		event->Emit(std::forward<Args>(args)...);
	}

}