#pragma once

#include "Display/Layout/Layout.h"

//A Layout with a single widget slot, covering its entire screenspace
class SingleLayout : public Layout
{

public:

	SingleLayout() = default;
	~SingleLayout() override = default;

	SingleLayout(const SingleLayout&) = delete;
	SingleLayout(SingleLayout&&) noexcept = default;

	SingleLayout& operator=(const SingleLayout&) = delete;
	SingleLayout& operator=(SingleLayout&&) noexcept = default;


};

