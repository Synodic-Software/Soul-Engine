#pragma once

#include "Types.h"
#include "Core/Structure/ExternalStructure.h"

#include <vector>
#include <cassert>
#include <memory>

template<class T>
class ExternalVector : public ExternalStructure<T> {

	using size_type = size_t;

public:

	ExternalVector() = default;
	~ExternalVector() override = default;


};