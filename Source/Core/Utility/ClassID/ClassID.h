#pragma once

#include <atomic>

class ClassID
{

public:

	ClassID() = delete;

	template<typename... Type>
	static size_t Id() noexcept;


private:

	static std::atomic<size_t> counter_;

	template<typename...>
	static size_t GenerateID() noexcept;

};


template<typename... Type>
size_t ClassID::Id() noexcept {
	return GenerateID<std::decay_t<Type>...>();
}

template<typename...>
size_t ClassID::GenerateID() noexcept {
	static const size_t value = counter_.fetch_add(1);
	return value;
}
