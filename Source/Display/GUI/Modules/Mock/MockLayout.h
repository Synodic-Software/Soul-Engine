#pragma once

#include "GUI/Layout.h"


class MockLayout : public Layout
{

public:

	MockLayout() = default;
	~MockLayout() override = default;

	MockLayout(const MockLayout&) = delete;
	MockLayout(MockLayout&&) noexcept = default;

	MockLayout& operator=(const MockLayout&) = delete;
	MockLayout& operator=(MockLayout&&) noexcept = default;


};
