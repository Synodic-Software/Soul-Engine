#pragma once


template <class T, template<typename> class ClassType>
class CRTP
{

public:

	virtual ~CRTP() = default;

	//return the derived class
	T& Type();
	T const& Type() const;


private:

	CRTP() = default;

	friend ClassType<T>;


};

template <class T, template<typename> class ClassType>
T& CRTP<T, ClassType>::Type() {

	return static_cast<T&>(*this);

}

template <class T, template<typename> class ClassType>
T const& CRTP<T, ClassType>::Type() const {

	return static_cast<T const&>(*this);

}

