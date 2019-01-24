#pragma once

#include "Core/Utility/Types.h"

#include <functional>
#include <unordered_map>

using eventID = uint64;

class BaseEvent
{

public:

	BaseEvent() = default;
	virtual ~BaseEvent() = default;

	virtual void Remove(eventID) {}

};

//force specialization
template<typename>
class Event;

template<typename R, typename... Types>
class Event<R(Types...)> : public BaseEvent
{

public:

	using signature = std::function<R(Types...)>;

	Event() = default;
	~Event() = default;

	Event(const Event &) = delete;
	Event(Event &&) = default;

	Event & operator=(const Event &) = delete;
	Event & operator=(Event &&) = default;

	//stores the callable with quick lookup enabled by the ID
	void Listen(eventID, signature&&);
	void Remove(eventID) override;
	void RemoveAll() const;

	//calls all stored callables
	template<typename... Args>
	void Emit(Args&&... args);

private:

	//the hashmap of 
	mutable std::unordered_map<eventID, signature> listeners;

};


template<typename R, typename ... Types>
void Event<R(Types...)>::Listen(eventID id, signature&& fn) {
	listeners.insert(std::make_pair(id, fn));
}

template<typename R, typename ... Types>
void Event<R(Types...)>::Remove(eventID id) {
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