#pragma once

#include "Types.h"
#include "Core/Structure/ExternalStructure.h"
#include "Core/Utility/Exception/Exception.h"

#include <array>
#include <cassert>

template<class T, size_t N>
class ExternalArray final : public SparseStructure {

	using size_type = size_t;


public:

	ExternalArray() = default;
	~ExternalArray() override = default;


private:

	std::array<T, N> data_;


};