#pragma once

#include "Types.h"

#include <vector>
#include <cassert>


template <typename T>
class IntrusiveSparseSet{

public:

	IntrusiveSparseSet() = default;
	virtual ~IntrusiveSparseSet() = default;

	IntrusiveSparseSet(const IntrusiveSparseSet &) = delete;
	IntrusiveSparseSet(IntrusiveSparseSet &&) noexcept = default;

	IntrusiveSparseSet & operator=(const IntrusiveSparseSet &) = delete;
	IntrusiveSparseSet & operator=(IntrusiveSparseSet &&) noexcept = default;

	//operators
	T& operator [](uint i);


protected:

	std::vector<T> objects_;

};

template <typename T>
T& IntrusiveSparseSet<T>::operator [](uint i)
{
	assert(Find(i));
	return objects_[sparse_[i]];
}