#pragma once

#include "GUI/Widget.h"


class MockWidget : public Widget {

public:

	MockWidget() = default;
	~MockWidget() override = default;

	MockWidget(const MockWidget&) = delete;
	MockWidget(MockWidget&&) noexcept = default;

	MockWidget& operator=(const MockWidget&) = delete;
	MockWidget& operator=(MockWidget&&) noexcept = default;


};

