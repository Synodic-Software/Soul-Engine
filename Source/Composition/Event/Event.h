#pragma once

#include "Parallelism/Fiber/Scheduler.h"

#include "Core/Utility/Types.h"

#include <functional>
#include <unordered_map>

using EventID = uint64;

class BaseEvent
{

public:

	BaseEvent() = default;
	virtual ~BaseEvent() = default;

	virtual void Remove(EventID) {}

};

template<typename>
class Event;

template<typename R, typename... Types>
class Event<R(Types...)> : public BaseEvent
{

public:

	using signature = std::function<R(Types...)>;

	Event() = default;
	~Event() = default;

	//stores the callable with quick lookup enabled by the ID
	void Listen(EventID ID, signature&& fn);
	void Remove(EventID ID) override;
	void RemoveAll() const;

	//calls all stored callables
	template<typename... Args>
	void Emit(Args&&... args);

private:

	//the hashmap of 
	mutable std::unordered_map<EventID, signature> listeners;

};


template<typename R, typename ... Types>
void Event<R(Types...)>::Listen(EventID id, signature&& fn) {
	listeners.insert(std::make_pair(id, fn));
}

template<typename R, typename ... Types>
void Event<R(Types...)>::Remove(EventID id) {
	listeners.erase(id);
}

template<typename R, typename ... Types>
void Event<R(Types...)>::RemoveAll() const {
	listeners.clear();
}

template<typename R, typename ... Types>
template<typename... Args>
void Event<R(Types...)>::Emit(Args&&... args) {
	for (const auto&[key, value] : listeners) {
		std::invoke(value, std::forward<Args>(args)...);
	}
}