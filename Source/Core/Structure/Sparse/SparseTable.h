#pragma once

#include "Types.h"
#include "Core/Structure/SparseStructure.h"
#include "SparseBitMap.h"
#include "Core/Utility/Exception/Exception.h"

#include <vector>
#include <cassert>

template<class T>
class SparseTableIterator {

	using iterator = SparseTableIterator;
	using value_type = T;
	using pointer = T*;
	using reference = T&;
	using size_type = size_t;

public:

	// TODO: implement

private:

	// TODO: implement

};

/* Maps an input key by hash into contiguous index. */
template<class Key, class Hash = std::hash<Key>>
class SparseTable : public SparseStructure {

	using size_type = size_t;
	using iterator = SparseTableIterator<Key>;

	static constexpr uint8 blockSize_ = 8;


public:

	SparseTable();
	~SparseTable() override = default;

	virtual void Erase(const Key& key);
	iterator Find(const Key& key);

	iterator end() noexcept;

protected:

	static constexpr Hash hasher_ = {};

	double minLoadFactor_;
	double maxLoadFactor_;

	std::vector<SparseBitMap<size_t, blockSize_>> blocks_;


};

template<class Key, class Hash>
SparseTable<Key, Hash>::SparseTable(): 
	minLoadFactor_(0.20), 
	maxLoadFactor_(0.80), 
	blocks_(4)
{
}

template<class Key, class Hash>
void SparseTable<Key, Hash>::Erase(const Key& key)
{

	throw NotImplemented();

}

template<class Key, class Hash>
typename SparseTable<Key, Hash>::iterator SparseTable<Key, Hash>::Find(const Key& key)
{

	throw NotImplemented();

}

template<class Key, class Hash>
typename SparseTable<Key, Hash>::iterator SparseTable<Key, Hash>::end() noexcept
{

	throw NotImplemented();

}