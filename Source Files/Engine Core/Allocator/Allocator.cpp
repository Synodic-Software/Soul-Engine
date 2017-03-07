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

template<Class T> T* Allocator::allocateNew(Allocator& allocator) {
	return new (allocator.allocate(sizeof(T), __alignof(T))) T;
}

template<Class T> T* Allocator::allocateNew(Allocator& allocator, const T& t) {
	return new (allocator.allocate(sizeof(T), __alignof(T))) T(t);
}

template<Class T> T* Allocator::allocateNewArr(Allocator& allocator, size_t length) {
	assert(length != 0);
	T* arr = (T*) allocator.allocate(sizeof(T)*length, __alignof(T));
	return p;
}

template<Class T> void deallocateItem(Allocator& allocator, T& object) {
	object.~T();
	allocator.deallocate(&object);
}

template<Class T> void deallocateArr(Allocator& allocator, T* arr) {
	assert(arr != nullptr);
	allocator.deallocate(arr);
}