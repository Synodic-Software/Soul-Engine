#include "Widget.h"

Widget::Widget() {
	DirtyFlag(true); //nothing is rendered, so draw
}

bool Widget::DirtyFlag() const {
	return flags_[0];
}

void Widget::DirtyFlag(const bool on) {
	flags_[0] = on;
}