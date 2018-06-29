#pragma once

#include "Display/Window/SoulWindow.h"
#include "Display/Window/DisplayManager.h"

class DesktopWindow : public SoulWindow 
{
public:

	DesktopWindow(WindowParameters&, void*, void*);
	~DesktopWindow();


	void Draw();

	void Refresh() override;
	void Close() override;
	void Resize(int, int) override;
	void PositionUpdate(int,int) override;
	void SetLayout(Layout*) override;

protected:

private:
};