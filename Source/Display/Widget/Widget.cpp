#include "Widget.h"

bool Widget::DirtyFlag() const {
	return flags_[0];
}

void Widget::DirtyFlag(bool on) {
	flags_[0] = on;
}