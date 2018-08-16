#pragma once

#include "Composition/Component/Component.h"

#include <unordered_map>

//Convenience class masking static memory for thread local storage on a per object basis
template<typename T>
class ThreadLocal : public Component<ThreadLocal<T>> {

public:

	ThreadLocal();
	~ThreadLocal() override = default;

	ThreadLocal(const ThreadLocal&) = delete;
	ThreadLocal(ThreadLocal&& o) noexcept = default;

	ThreadLocal& operator=(const ThreadLocal&) = delete;
	ThreadLocal& operator=(ThreadLocal&& other) noexcept = default;

	ThreadLocal<T>(const T&);
	ThreadLocal<T>& operator= (const T&);
	operator T();

	void Terminate() override;

private:

	int id_;

	static int counter_;
	static thread_local std::unordered_map<int, T> objectMap_;

};

template<typename T>
ThreadLocal<T>::ThreadLocal():
	id_(counter_++)
{
}

template<typename T>
void ThreadLocal<T>::Terminate() {

	objectMap_.erase(id_);

}

template<typename T>
ThreadLocal<T>::ThreadLocal(const T& value):
	ThreadLocal()
{

	objectMap_[id_] = value;

}

template<typename T>
ThreadLocal<T>& ThreadLocal<T>::operator= (const T& value) {

	objectMap_[id_] = value;
	return *this;

}

template<typename T>
ThreadLocal<T>::operator T() {

	return objectMap_[id_];

}

template<typename T>
int ThreadLocal<T>::counter_ = 0;

template<typename T>
thread_local std::unordered_map<int, T> ThreadLocal<T>::objectMap_;