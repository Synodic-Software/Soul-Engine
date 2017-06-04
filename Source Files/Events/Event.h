#pragma once

#include "Multithreading/Scheduler.h"

#include <list>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

class BaseEvent
{
public:
	BaseEvent() {};
	virtual ~BaseEvent() {}
};

template<typename... Args>
class Event : public BaseEvent
{

public:

	using signature = std::function<void(Args...)>;

	Event() {}
	Event(signature f) {}
	~Event() {}



	//Listener functions
	template<typename T>
	void Listen(int ID, T *instance, void(T::*func)(Args...) const) {
		return Listen(ID, [=](Args... args) {
			(instance->*func)(args...);
		});
	}

	template<typename T>
	void Listen(int ID, T *instance, void(T::*func)(Args...)) {
		return Listen(ID, [=](Args... args) {
			(instance->*func)(args...);
		});
	}

	void Listen(int ID, const signature& fn) {
		listeners.insert(std::make_pair(ID, fn));
	}

	void Listen(int ID, signature&& fn) {
		listeners.insert(std::make_pair(ID, fn));
	}

	//Removal functions
	void Remove(int ID) {
		listeners.erase(ID);
	}
	void RemoveAll() {
		listeners.clear();
	}
	
	//Notify functions
	void Notify(int ID,Args... args) {

	}
	void NotifyAll(Args... args) {
		for (auto itr : listeners) {
			itr.second(args...);
		}
	}

protected:

private:

	mutable std::map<int, signature> listeners;

};
