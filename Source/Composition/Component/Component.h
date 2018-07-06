#pragma once

#include <atomic>


class Component
{

public:

	Component() = default;

private:

	static std::atomic<size_t> counter_;

	size_t id_;


};

//Component::Component() :
//	id_(counter_.fetch_add(1)) {
//
//}
