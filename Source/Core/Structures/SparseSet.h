#pragma once

#include "AbstractSparseSet.h"

template <typename T>
class SparseSet final : public AbstractSparseSet {

public:

	SparseSet() = default;
	~SparseSet() override = default;

	SparseSet(const SparseSet &) = delete;
	SparseSet(SparseSet &&) = default;

	SparseSet & operator=(const SparseSet &) = delete;
	SparseSet & operator=(SparseSet &&) = default;


private:

	std::vector<size_t> objects_;

};


