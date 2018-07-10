#pragma once

#include "glm/glm.hpp"

#include <bitset>

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

	//sub-pixel values
	glm::dvec2 size_;
	glm::dvec2 position_; //upper left position

	//Gets
	bool DirtyFlag() const; //flag 0

	//Sets
	void DirtyFlag(bool);


private:

	std::bitset<1> flags_;


};

