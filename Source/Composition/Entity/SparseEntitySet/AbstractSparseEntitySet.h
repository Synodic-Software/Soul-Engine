#pragma once

#include "Core/Structures/AbstractSparseSet.h"
#include "Composition/Entity/Entity.h"

#include <vector>

//empty type base implementation
class AbstractSparseEntitySet: public virtual AbstractSparseSet {

public:

	AbstractSparseEntitySet() = default;
	virtual  ~AbstractSparseEntitySet() = default;

	AbstractSparseEntitySet(const AbstractSparseEntitySet &) = delete;
	AbstractSparseEntitySet(AbstractSparseEntitySet &&) noexcept = default;

	AbstractSparseEntitySet & operator=(const AbstractSparseEntitySet &) = delete;
	AbstractSparseEntitySet & operator=(AbstractSparseEntitySet &&) noexcept = default;

	//operators

	//get/set
	void Remove(uint) override = 0;

protected:

	std::vector<Entity> dense_;

};
