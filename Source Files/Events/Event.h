#pragma once

#include "Multithreading/Scheduler.h"

#include "Metrics.h"

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

	virtual void Remove(uint) {};
};

template<typename... Args>
class Event : public BaseEvent
{

public:

	using signature = std::function<void(Args...)>;

	Event() {}
	~Event() {}

	void Listen(int ID, const signature& fn) {
		listeners.insert(std::make_pair(ID, fn));
	}

	void Listen(int ID, signature&& fn) {
		listeners.insert(std::make_pair(ID, fn));
	}

	//Removal functions
	virtual void Remove(int ID) {
		listeners.erase(ID);
	}

	void RemoveAll() {
		listeners.clear();
	}
	
	void Emit(Args... args) {
		for (auto itr : listeners) {
			itr.second(args...);
		}
	}

protected:

private:

	mutable std::map<int, signature> listeners;

};
