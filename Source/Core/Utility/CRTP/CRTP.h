#pragma once

template <class T, template<typename> class type>
class CRTP
{

public:

	CRTP() = delete;

	T& Type();
	T const& Type() const;


private:
	
	friend type<T>;

};

template <class T, template<typename> class type>
T& CRTP<T, type>::Type() {

	return static_cast<T&>(*this);

}

template <class T, template<typename> class type>
T const& CRTP<T, type>::Type() const {

	return static_cast<T const&>(*this);

}