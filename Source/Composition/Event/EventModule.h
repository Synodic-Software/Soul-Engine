#pragma once

#include "Core/Utility/HashString/HashString.h"

#include <unordered_map>
#include <memory>

//TODO replace with Callable Concept c++20
#include <boost/callable_traits.hpp>

//signal and slots system
class EventModule {

public:

	EventModule() = default;
	~EventModule() = default;

	EventModule(const EventModule&) = delete;
	EventModule(EventModule&&) = default;

	EventModule & operator=(const EventModule&) = delete;
	EventModule & operator=(EventModule&&) = default;

	//event functions
	template<typename Fn>
	uint64 Listen(HashString::HashType, HashString::HashType, Fn&&);

	template <typename... Args>
	void Emit(HashString::HashType, HashString::HashType, Args&&...);

	virtual void Remove(HashString::HashType, HashString::HashType, uint64) = 0;
	virtual void Remove(HashString::HashType, HashString::HashType) = 0;
	virtual void Remove(HashString::HashType) = 0;

	// Factory
	static std::unique_ptr<EventModule> CreateModule();


};

template<typename Fn>
uint64 EventModule::Listen(HashString::HashType channel, HashString::HashType name, Fn&& func)
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
void EventModule::Emit(HashString::HashType channel, HashString::HashType name, Args&&... args)
{

	//TODO cast return type
	if (auto event = static_cast<Event<void(decltype(args)...)>*>(eventMap_[channel][name].get()); event)
	{
		event->Emit(std::forward<Args>(args)...);
	}

}