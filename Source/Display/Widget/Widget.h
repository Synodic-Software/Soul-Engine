#pragma once

#include "Composition/Entity/Entity.h"
#include "glm/glm.hpp"

class Widget
{

public:

	Widget() = default;
	virtual ~Widget() = default;

	Widget(const Widget&) = delete;
	Widget(Widget&&) noexcept = default;

	Widget& operator=(const Widget&) = delete;
	Widget& operator=(Widget&&) noexcept = default;

protected:

	Entity widgetJob_;

	glm::dvec2 size_;
	glm::dvec2 position_; //upper left position


};

