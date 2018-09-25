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

	std::size_t begin_;
	std::size_t end_;
	std::size_t size_;

	std::array<T, Capacity> data_;

	void Push_();

};

template <typename T, std::size_t Capacity>
RingBuffer<T, Capacity>::RingBuffer() :
	begin_(1),
	end_(0),
	size_(Capacity),
	data_()
{

}


template <typename T, std::size_t Capacity>
void RingBuffer<T, Capacity>::Push(const T& value) {

	data_[end_] = value;
	Push_();

}

template <typename T, std::size_t Capacity>
void RingBuffer<T, Capacity>::Push(T&& value) {

	data_[end_] = std::move(value);
	Push_();

}

template <typename T, std::size_t Capacity>
typename RingBuffer<T, Capacity>::reference RingBuffer<T, Capacity>::operator[](const std::size_t i)
{
	const auto index = (i + begin_) % size_;
	return data_[index];
}

template <typename T, std::size_t Capacity>
void RingBuffer<T, Capacity>::Push_() {

	++end_;
	if (end_ == size_)
	{
		end_ = 0;
	}

	if (size_ != data_.size())
	{
		++size_;
	}

}