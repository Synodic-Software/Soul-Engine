#pragma once

class Framebuffer {

public:

	Framebuffer() = default;
	virtual ~Framebuffer() = default;

	Framebuffer(const Framebuffer&) = delete;
	Framebuffer(Framebuffer&& o) noexcept = default;

	Framebuffer& operator=(const Framebuffer&) = delete;
	Framebuffer& operator=(Framebuffer&& other) noexcept = default;

};
