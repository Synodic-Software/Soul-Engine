#pragma once
#include "Display\Widget\Widget.h"

#include "Metrics.h"
#include <list>
#include <memory>

class Layout : public Widget
{

public:

	Layout();
	~Layout();

	virtual void Draw();

protected:

	std::list<std::unique_ptr<Widget> > widgets;

private:
};

