//TODO: finish implementation
#include<"Allocator.h">
Allocator::Allocator(size_t size, void* start) {
	_start = start;
	_capacity = size;
	_usedMem = 0;
	_numAllocs = 0;
}

Allocator::~Allocator() {
	assert(usedMem == 0 && numAllocs == 0);
	_start = nullptr;
	_capacity = 0;
}