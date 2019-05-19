#pragma once

#include "SparseVector.h"

#include "Types.h"
#include "Core/Structure/SparseStructure.h"
#include "SparseTable.h"
#include "Core/Utility/Exception/Exception.h"

#include <vector>
#include <cassert>


template<class Key, class Value, class Hash = std::hash<Key>>
class SparseHashMap final : public SparseTable<Key, Hash> {

	using size_type = size_t;

	static constexpr uint8 blockSize_ = 8;

public:

	SparseHashMap() = default;
	~SparseHashMap() override = default;

	Value& operator[](const Key&);

	template<typename... Args>
	std::pair<Value&, bool> Emplace(const Key&, Args&&...);

	void Clear();
	void Erase(const Key& key) override;

private:

	/*
	 * Inserts a hash into the the vector of blocks. A precondition is that there is room for the insertion
	 *
	 * @param	key	   	The key.
	 * @param	indirection	The indirection value pointing to the insert.
	 *
	 * @returns	A pair;
	 */

	std::pair<size_t&, bool> Insert(const Key& key, size_t indirection);

	bool Full() const;
	size_t ReduceHash(size_t hash) const;
	void Rehash(size_t);

	std::vector<Key> keys_;
	std::vector<Value> values_;


};

template<class Key, class Value, class Hash>
Value& SparseHashMap<Key, Value, Hash>::operator[](const Key& key)
{

	//TODO: should not be a part of []
	if (Full()) {

		Rehash(blocks_.size() * 2);
	}

	auto& [indirection, inserted] = Insert(key, values_.size());

	if (inserted) {

		keys_.push_back(key);
		return values_.emplace_back();

	}
	else {

		return values_[indirection];

	}

}

template<class Key, class Value, class Hash>
template<typename... Args>
std::pair<Value&, bool> SparseHashMap<Key, Value, Hash>::Emplace(const Key& key, Args&&... args)
{

	if (Full()) {

		Rehash(blocks_.size() * 2);

	}

	size_t hash = hasher_(key);
 	size_t index = ReduceHash(hash);

	for (size_t i=0; true; ++i) {

		size_t blockIndex = index / blockSize_;
		uint8 innerIndex = index % blockSize_;

		auto& block = blocks_[blockIndex];

		if (!block.Test(innerIndex)) {

			size_t indirection = values_.size();
			auto& returnValue = values_.emplace_back(std::forward<Args>(args)...);
			keys_.push_back(key);

			block.Insert(innerIndex, indirection);
			return {returnValue, true};

		}

		if (size_t indirection = block[innerIndex]; hasher_(keys_[indirection]) == hash) {

			return {values_[indirection], false};

		}

		//Search for a new spot
		index += ( hash + i * i ) % values_.size();

		//Wrap to the blocksize
		index &= blocks_.size() - 1;

	}

	//TODO: should never reach here. Error handling? Dummy return?
	throw NotImplemented();

}

template<class Key, class Value, class Hash>
void SparseHashMap<Key, Value, Hash>::Clear()
{

	blocks_.clear();
	keys_.clear();
	values_.clear();

}

template<class Key, class Value, class Hash>
std::pair<size_t&, bool> SparseHashMap<Key, Value, Hash>::Insert(const Key& key, size_t indirection) {

	size_t hash = hasher_(key);
	size_t index = ReduceHash(hash);

	for (size_t i=0; true; ++i) {

		size_t blockIndex = index / blockSize_;
		uint8 innerIndex = index % blockSize_;

		auto& block = blocks_[blockIndex];

		if (!block.Test(innerIndex)) {

			block.Insert(innerIndex, indirection);
			return {indirection, true};
		}

		if (size_t storedIndirection = block[innerIndex]; hasher_(keys_[indirection]) == hash) {

			return {storedIndirection, false};
		}

		// Search for a new spot
		index += ( hash + i * i ) % values_.size();

		// Wrap to the blocksize
		index &= blocks_.size() * blockSize_ - 1;
	}

	//TODO: should never reach here. Error handling? Dummy return?
	throw NotImplemented();

}

template<class Key, class Value, class Hash>
bool SparseHashMap<Key, Value, Hash>::Full() const {

	return values_.size() + 1 > blocks_.size() * blockSize_ * maxLoadFactor_;

}

template<class Key, class Value, class Hash>
size_t SparseHashMap<Key, Value, Hash>::ReduceHash(size_t hash) const
{

	hash ^= hash >> 7;
	hash *= 11400714819323198485llu >> 7;
    return hash & (blocks_.size() * blockSize_ - 1);

}

template<class Key, class Value, class Hash>
void SparseHashMap<Key, Value, Hash>::Rehash(size_t newBlockSize) {

	for (auto& block : blocks_) {

		block.Clear();

	}

	blocks_.resize(std::max(newBlockSize, static_cast<size_t>(4)));

	for (int i = 0; i < keys_.size(); ++i) {

		Insert(keys_[i], i);

	}


}

template<class Key, class Value, class Hash>
void SparseHashMap<Key, Value, Hash>::Erase(const Key& key) {

	//auto indirectionIterator = SparseTable<Key, Hash>::Find(key);

	//if(indirectionIterator != SparseTable<Key, Hash>::end()){

	//	if (values_.size() > 1)
	//	{
	//		//TODO: iterator swap
	//		std::swap(keys_[indirection], *keys.back());
	//		std::swap(values_[indirection], *values_.back());

	//		SparseTable<Key, Hash>::Erase(keys_.back());
	//		keys_.pop_back();
	//		values_.pop_back();

	//		SparseTable<Key, Hash>::Insert(keys_[indirection], indirection);

	//	}
	//	else
	//	{

	//		keys_.clear();
	//		values_.clear();
	//		blocks_.Clear();

	//	}

	//	SparseTable<Key, Hash>::Erase(key);

	//}

	throw NotImplemented();

}