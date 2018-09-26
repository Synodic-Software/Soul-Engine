#pragma once

template <template<typename, typename...> class Fn>
class Task{

public:

	Task() = default;

	template <typename Callable>
	Task(Callable&&);

	Task(const Task&) = delete;
	Task(Task&& o) noexcept = delete;

	Task& operator=(const Task&) = delete;
	Task& operator=(Task&& other) noexcept = delete;

private:


};

template <template<typename, typename...> class Fn>
template <typename Callable>
Task<Fn>::Task(Callable&&) {
	
}