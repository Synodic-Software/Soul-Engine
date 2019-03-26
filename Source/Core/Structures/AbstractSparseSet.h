#pragma once

#include "Types.h"

#include <vector>

//empty type base implementation
class AbstractSparseSet {

public:

	AbstractSparseSet() = default;
	virtual  ~AbstractSparseSet() = default;

	AbstractSparseSet(const AbstractSparseSet &) = delete;
	AbstractSparseSet(AbstractSparseSet &&) noexcept = default;

	AbstractSparseSet & operator=(const AbstractSparseSet &) = delete;
	AbstractSparseSet & operator=(AbstractSparseSet &&) noexcept = default;

	//operators

	//get/set
	virtual void Remove(uint) = 0;

	bool Find(size_t);

protected:

	static constexpr uint emptyValue = uint(-1);

	std::vector<uint> sparse_;

};