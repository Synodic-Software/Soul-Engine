#pragma once

template<class T>
class View{

	// TODO: C++20 Concepts
	static_assert(std::is_base_of<Contiguous, T>::value,
		"The T parameter must be a subclass of Contiguous");


public:

	View() = default;
	virtual ~View() = default;


};