#pragma once

#include <vector>

//empty type base implementation
class AbstractSparseSet {

public:

	AbstractSparseSet() = default;
	virtual  ~AbstractSparseSet() = default;

	AbstractSparseSet(const AbstractSparseSet &) = delete;
	AbstractSparseSet(AbstractSparseSet &&) = default;

	AbstractSparseSet & operator=(const AbstractSparseSet &) = delete;
	AbstractSparseSet & operator=(AbstractSparseSet &&) = default;


protected:

	std::vector<size_t> dense_;
	std::vector<size_t> sparse_;

};
