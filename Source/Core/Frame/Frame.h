#pragma once

#include <cstddef>

class Frame {

public:

	Frame();
	Frame(const Frame&, const Frame&);
	~Frame() = default;

	Frame(const Frame&) = delete;
	Frame(Frame&&) noexcept = default;

	Frame& operator=(const Frame&) = delete;
	Frame& operator=(Frame&&) noexcept = default;

private:

	std::size_t id_;

};
