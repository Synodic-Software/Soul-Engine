#include "Layout.h"

Layout::Layout()
{
}

Layout::~Layout()
{
}

void Layout::Draw() {
	for (auto& wid : widgets) {
		wid->Draw();
	}
}
