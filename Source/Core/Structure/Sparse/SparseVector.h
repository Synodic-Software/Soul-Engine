#pragma once

#include "Types.h"
#include "Core/Structure/SparseStructure.h"
#include "SparseBitMap.h"
#include "Core/Utility/Exception/Exception.h"

#include <vector>
#include <cassert>
#include <memory>

template<class T, size_t N>
class SparseVector final : public SparseStructure {

	using size_type = size_t;

	static constexpr size_type groupSize_ = 32;


public:

	SparseVector() = default;
	~SparseVector() override = default;

	T& operator[](size_type);
	bool Test(size_type) const;


private:

	std::unique_ptr<SparseBitMap<T, groupSize_>> data_;


};

template<class T, size_t N>
T& SparseVector<T, N>::operator[](size_type index)
{

	throw NotImplemented();

}

template<class T, size_t N>
bool SparseVector<T, N>::Test(size_type index) const
{

	throw NotImplemented();

}