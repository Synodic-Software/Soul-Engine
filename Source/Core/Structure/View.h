#pragma once

#include <functional>
#include <optional>

template<class T>
class View<T> {

	using size_type = size_t;
	using function_type = std::function<size_type(size_type)>;

public:

	constexpr View();
	constexpr View(T* ptr, size_type count);
	virtual ~View() = default;

	T& operator[](size_type);

	void Map(function_type&& function);


private:

	// TODO: std::unique_function C++20
	std::optional<function_type> indexMap_;

	//TODO: replace with std::span C++20
	const T* data_;
	const size_type size_;


};

template<class T>
constexpr View<T>::View(): data_(nullptr), size_(0)
{
}

template<class T>
constexpr View<T>::View(T* ptr, size_type count): data_(ptr), size_(count)
{
}

template<class T>
T& View<T>::operator[](size_type index)
{
	if (!indexMap_.has_value()) {
		return data_[index];
	}
	
	return data_[*indexMap_(index)];

}

template<class T>
void View<T>::Map(function_type&& function)
{

	indexMap_.emplace(std::move(myFunction))

}