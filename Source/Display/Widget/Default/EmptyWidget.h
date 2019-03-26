#pragma once

#include "Display/Widget/Widget.h"


class EmptyWidget : public Widget
{

public:

	EmptyWidget() = default;
	~EmptyWidget() = default;

	EmptyWidget(const EmptyWidget&) = delete;
	EmptyWidget(EmptyWidget&&) noexcept = default;

	EmptyWidget& operator=(const EmptyWidget&) = delete;
	EmptyWidget& operator=(EmptyWidget&&) noexcept = default;


};

