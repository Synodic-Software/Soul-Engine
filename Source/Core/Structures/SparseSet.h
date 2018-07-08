#pragma once

#include "AbstractSparseSet.h"
#include "Core/Utility/Types.h"

#include <cassert>


template <typename T>
class SparseSet : public virtual AbstractSparseSet {

public:

	SparseSet() = default;
	virtual ~SparseSet() = default;

	SparseSet(const SparseSet &) = delete;
	SparseSet(SparseSet &&) noexcept = default;

	SparseSet & operator=(const SparseSet &) = delete;
	SparseSet & operator=(SparseSet &&) noexcept = default;

	//operators
	T& operator [](uint i);

	//set/get


protected:

	std::vector<T> objects_;

};

template <typename T>
T& SparseSet<T>::operator [](uint i)
{
	assert(Find(i));
	return objects_[sparse_[i]];
}