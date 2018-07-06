#pragma once

#include "Event.h"
#include "Parallelism/Fiber/Scheduler.h"
#include "Core/Utility/HashString/HashString.h"

#include <unordered_map>
#include <memory>

//TODO replace with Callable Concept c++20
#include <boost/callable_traits.hpp>

//signal and slots system
class EventManager final {

public:

	//construction and assignment
	EventManager();
	~EventManager() = default;

	EventManager(const EventManager &) = delete;
	EventManager(EventManager &&) = default;

	EventManager & operator=(const EventManager &) = delete;
	EventManager & operator=(EventManager &&) = default;

	//event functions
	template<typename Fn>
	eventID Listen(HashString::HashType, HashString::HashType, Fn&&);

	template <typename... Args>
	void Emit(HashString::HashType, HashString::HashType, Args&&...);

	void Remove(HashString::HashType, HashString::HashType, eventID);
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
	eventID idCounter_;

};

template<typename Fn>
eventID EventManager::Listen(HashString::HashType channel, HashString::HashType name, Fn&& func)
{
	using type = boost::callable_traits::function_type_t<Fn>;

	auto& baseEventPtr = eventMap_[channel][name];
	if (!baseEventPtr.get()) {
		baseEventPtr = std::make_unique<Event<type>>();
	}

	auto event = static_cast<Event<type>*>(baseEventPtr.get());

	event->Listen(idCounter_, std::forward<Fn>(func));

	return idCounter_++;
}

template <typename... Args>
void EventManager::Emit(HashString::HashType channel, HashString::HashType name, Args&&... args)
{

	//TODO cast return type
	if (auto event = static_cast<Event<void(decltype(args)...)>*>(eventMap_[channel][name].get()); event)
	{
		event->Emit(std::forward<Args>(args)...);
	}

}