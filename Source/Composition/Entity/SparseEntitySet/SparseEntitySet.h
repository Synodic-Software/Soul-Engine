#pragma once

#include "AbstractSparseEntitySet.h"
#include "Core/Structures/SparseSet.h"

#include <cassert>

template <typename T>
class SparseEntitySet : public AbstractSparseEntitySet, public SparseSet<T> {

public:

	SparseEntitySet() = default;
	~SparseEntitySet() override = default;

	SparseEntitySet(const SparseEntitySet &) = delete;
	SparseEntitySet(SparseEntitySet &&) = default;

	SparseEntitySet & operator=(const SparseEntitySet &) = delete;
	SparseEntitySet & operator=(SparseEntitySet &&) = default;

	//operators

	//get/set
	void Remove(uint) override;

	template<typename... Args>
	void Insert(Entity, Args&&...);

};

template <typename T>
void SparseEntitySet<T>::Remove(uint i) {
	//TODO implement
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
