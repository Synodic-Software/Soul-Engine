#pragma once

#include "Types.h"
#include "SparseStructure.h"
#include "SparseTable.h"
#include "Core/Utility/Exception/Exception.h"

#include <vector>
#include <cassert>


template<class Key, class Hash = std::hash<Key>>
class SparseHashSet final : public SparseTable<Key, Hash> {

	using size_type = size_t;

	static constexpr size_type blockSize_ = 8;

public:

	SparseHashSet() = default;
	~SparseHashSet() override = default;


private:

	std::vector<SparseBitMap<size_t, blockSize_>> sparseMapping_;
	std::vector<Key> keys_;


};