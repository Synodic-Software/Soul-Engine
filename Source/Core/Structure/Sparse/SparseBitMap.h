#pragma once

#include "Types.h"
#include "Core/Structure/SparseStructure.h"
#include "Core/Utility/Exception/Exception.h"

#include <vector>
#include <cassert>
#include <memory>
#include <type_traits>
#include <bitset>
#include <algorithm>


template<class T, uint8 N>
class SparseBitMapIterator {

	using iterator = SparseBitMapIterator;
	using value_type = T;
	using pointer = T*;
	using reference = T&;
	using size_type = size_t;


public:

	// TODO: implement


private:

	// TODO: implement


};


template<class T, uint8 N>
class SparseBitMap final : public SparseStructure {

	using size_type = uint8;
	using iterator = SparseBitMapIterator<T, N>;

	static_assert(N % 4 == 0, "N must be byte aligned.");


public:

	SparseBitMap() = default;
	~SparseBitMap() = default;

	T& operator[](size_type);
	bool Test(size_type) const;

	template<typename... Args>
	std::pair<T&, bool> Emplace(size_type, Args&&...);

	/*
	 * Inserts
	 *
	 * @param	index	The hash.
	 * @param	value	The indirection.
	 *
	 * @returns	A pair;
	 */

	std::pair<T&, bool> Insert(size_type index, const T& value);

	void Clear();

private:

	std::bitset<N> mapping_;
	std::vector<T> data_;


};

template<class T, uint8 N>
T& SparseBitMap<T, N>::operator[](size_type index)
{

	assert(mapping_.test(index));

	std::bitset<N> mask;
	mask = ~(~mask << index) & mapping_;

	//TODO: Replace with `popcount`
	size_type pos = static_cast<size_type>(mask.count()) - 1;

	assert(pos < data_.size());

	return data_[pos];

}

template<class T, uint8 N>
bool SparseBitMap<T, N>::Test(size_type index) const
{

	return mapping_.test(index);

}

template<class T, uint8 N>
template<typename... Args>
std::pair<T&, bool> SparseBitMap<T, N>::Emplace(size_type index, Args&&... args)
{

	if (Test(index)) {
		return {operator[](index), false};
	}

	std::bitset<N> mask;
	mask = ~(~mask << index) & mapping_;

	// TODO: Replace with `popcount`
	size_type pos = static_cast<size_type>(mask.count());

	auto iter = data_.emplace(data_.begin() + pos, std::forward<Args>(args)...);

	return {*iter, true};
}

template<class T, uint8 N>
std::pair<T&, bool> SparseBitMap<T, N>::Insert(size_type index, const T& value)
{

	if (Test(index)) {
		return {operator[](index), false};
	}

	std::bitset<N> mask;
	mask = ~(~mask << index) & mapping_;

	// TODO: Replace with `popcount`
	size_type pos = static_cast<size_type>(mask.count());

	auto iter = data_.insert(data_.begin() + pos, value);

	return {*iter, true};

}

template<class T, uint8 N>
void SparseBitMap<T, N>::Clear()
{

	mapping_.reset();
	data_.resize(0);

}