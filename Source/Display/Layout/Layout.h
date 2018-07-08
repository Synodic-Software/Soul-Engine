#pragma once

#include "Display/Widget/Widget.h"


class Layout : public Widget
{

public:

	Layout() = default;
	virtual ~Layout() = default;

	Layout(const Layout&) = delete;
	Layout(Layout&&) noexcept = default;

	Layout& operator=(const Layout&) = delete;
	Layout& operator=(Layout&&) noexcept = default;


};

