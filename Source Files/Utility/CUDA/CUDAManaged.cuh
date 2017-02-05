#pragma once

class Managed
{
public:
	void *operator new(size_t len);
	void operator delete(void *ptr);
};
