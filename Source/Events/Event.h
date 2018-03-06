#pragma once

#include "Multithreading/Scheduler.h"

#include "Metrics.h"

#include <list>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>
#include <unordered_map>

/* A base event. */
class BaseEvent
{
public:
	/* Default constructor. */
	BaseEvent() {};
	/* Destructor. */
	virtual ~BaseEvent() {}

	/*
	 *    Removes the given parameter1.
	 *    @param	parameter1	The parameter 1 to remove.
	 */

	virtual void Remove(uint) {}
};

template<typename... Args>
/* An event. */
class Event : public BaseEvent
{

public:

	/* The signature */
	using signature = std::function<void(Args...)>;

	/* Default constructor. */
	Event() {}
	/* Destructor. */
	~Event() {}

	/*
	 *    Listens.
	 *    @param	ID	The identifier.
	 *    @param	fn	The function.
	 */

	void Listen(int ID, const signature& fn) {
		listeners.insert(std::make_pair(ID, fn));
	}

	/*
	 *    Listens.
	 *    @param 		 	ID	The identifier.
	 *    @param [in,out]	fn	The function.
	 */

	void Listen(int ID, signature&& fn) {
		listeners.insert(std::make_pair(ID, fn));
	}

	/*
	 *    Removal functions.
	 *    @param	ID	The Identifier to remove.
	 */

	virtual void Remove(int ID) {
		listeners.erase(ID);
	}

	/* Removes all. */
	void RemoveAll() {
		listeners.clear();
	}

	/*
	 *    Emits the given arguments.
	 *    @param	args	Variable arguments providing the arguments.
	 */

	void Emit(Args... args) {
		for (auto itr : listeners) {
			itr.second(args...);
		}
	}

protected:

private:

	/*
	 *    Gets the listeners.
	 *    @return	The listeners.
	 */

	mutable std::unordered_map<int, signature> listeners;

};
