#pragma once

#include "AbstractSparseEntitySet.h"
#include "Core/Structures/SparseSet.h"

#include <cassert>

template <typename T>
class SparseEntitySet : public AbstractSparseEntitySet, public SparseSet<T> {

public:

	SparseEntitySet() = default;
	~SparseEntitySet() override;

	SparseEntitySet(const SparseEntitySet &) = delete;
	SparseEntitySet(SparseEntitySet &&) = default;

	SparseEntitySet & operator=(const SparseEntitySet &) = delete;
	SparseEntitySet & operator=(SparseEntitySet &&) = default;

	//operators

	//get/set
	void Remove(uint) override;

	template<typename... Args>
	void Insert(Entity, Args&&...);

	template<typename Derived, typename... Args>
	void Insert(Entity, Args&&...);

	void Clear();

};

template <typename T>
SparseEntitySet<T>::~SparseEntitySet() {

	for (auto& object : SparseSet<T>::objects_) {
		object.Terminate();
	}

}

template <typename T>
void SparseEntitySet<T>::Remove(uint entityID) {

	if (Find(entityID)) {

		//move last element to deleted position
		auto tmp = std::move(SparseSet<T>::objects_.back());
		SparseSet<T>::objects_.pop_back();

		(*this)[entityID] = std::move(tmp);


		//get
		const Entity tmp2 = dense_.back();
		auto& tmp2Ptr = sparse_[entityID];

		//swap
		dense_[tmp2Ptr] = tmp2;
		sparse_[tmp2.GetId()] = tmp2Ptr;

		//remove
		tmp2Ptr = emptyValue;
		dense_.pop_back();
	}

}

template <typename T>
template<typename... Args>
void SparseEntitySet<T>::Insert(Entity entity, Args&&... args) {

	const auto entityID = entity.GetId();

	assert(!Find(entityID));

	if (entityID >= sparse_.size()) {
		sparse_.resize(entityID + 1, emptyValue);
	}

	sparse_[entityID] = dense_.size();
	dense_.push_back(entity);

	SparseSet<T>::objects_.emplace_back(std::forward<Args>(args)...);

}

template <typename T>
void SparseEntitySet<T>::Clear() {

	//TODO: abstract into parent templates

	sparse_.clear();
	dense_.clear();

	for (auto& object: SparseSet<T>::objects_) {
		object.Terminate();
	}

	SparseSet<T>::objects_.clear();

}
