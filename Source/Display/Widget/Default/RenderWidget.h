#pragma once

#include "Display/Widget/Widget.h"


class RenderWidget : public Widget
{

public:

	RenderWidget() = default;
	~RenderWidget() = default;

	RenderWidget(const RenderWidget&) = delete;
	RenderWidget(RenderWidget&&) noexcept = default;

	RenderWidget& operator=(const RenderWidget&) = delete;
	RenderWidget& operator=(RenderWidget&&) noexcept = default;

};

