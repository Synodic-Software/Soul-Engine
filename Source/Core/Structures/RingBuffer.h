#pragma once

#include <array>

template <typename T, std::size_t Capacity>
class RingBuffer
{

public:

	using reference = T&;

	RingBuffer();
	~RingBuffer() = default;

	RingBuffer(const RingBuffer&) = delete;
	RingBuffer(RingBuffer&& o) noexcept = delete;

	RingBuffer& operator=(const RingBuffer&) = delete;
	RingBuffer& operator=(RingBuffer&& other) noexcept = delete;

	reference operator[](std::size_t i);

	void Push(const T&);
	void Push(T&&);

private:

	std::size_t front_;
	std::size_t size_;

	std::array<T, Capacity> data_;

	void Push_();

};

template <typename T, std::size_t Capacity>
RingBuffer<T, Capacity>::RingBuffer() :
	front_(0),
	size_(Capacity),
	data_()
{

}


template <typename T, std::size_t Capacity>
void RingBuffer<T, Capacity>::Push(const T& value) {

	Push_();
	data_[front_] = value;
		
}

template <typename T, std::size_t Capacity>
void RingBuffer<T, Capacity>::Push(T&& value) {

	Push_();
	data_[front_] = std::move(value);

}

template <typename T, std::size_t Capacity>
typename RingBuffer<T, Capacity>::reference RingBuffer<T, Capacity>::operator[](const std::size_t i)
{
	const auto index = (i + front_) % size_;
	return data_[index];
}

template <typename T, std::size_t Capacity>
void RingBuffer<T, Capacity>::Push_() {

	front_ = (front_+ size_ -1) % size_;

}