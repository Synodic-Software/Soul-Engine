#include "Window.h"

#include "Display/Layout/Default/SingleLayout.h"
#include "Display/Widget/Default/EmptyWidget.h"

Window::Window(WindowParameters& params, EntityManager& entityManager) :
	context_(nullptr),
	layout_(std::make_unique<SingleLayout>()),
	windowParams_(params)
{

	auto& wid = layout_->AddWidget<EmptyWidget>();

}

std::any& Window::GetContext() {
	return context_;
}
