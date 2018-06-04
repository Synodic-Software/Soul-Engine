#pragma once

#include <functional>

template<typename T>
class Property {

public:

	Property(const T&);
	~Property() = default;

	Property& operator=(const T&);

	operator T() const;

	void AddCallback(const std::function<void(T)>&);

private:

	T data;
	std::vector<std::function<void(T)>> callbacks;

};

template<typename T>
Property<T>::Property(const T& value) {
	data = value;
}


template<typename T>
Property<T>& Property<T>::operator=(const T& value){

	data = value;

	for(const auto& callback : callbacks) {
		callback(data);
	}

	return *this;

}

template<typename T>
Property<T>::operator T() const {

	return data;

}

template<typename T>
void Property<T>::AddCallback(const std::function<void(T)>& fn) {

	callbacks.push_back(fn);

}