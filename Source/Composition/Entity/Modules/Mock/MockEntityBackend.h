#pragma once


class MockEntityBackend
{

	friend class Entity;

public:

	//construction and assignment
	MockEntityBackend() = default;
	~MockEntityBackend() = default;

	MockEntityBackend(const MockEntityBackend &) = delete;
	MockEntityBackend(MockEntityBackend&&) = delete;

	MockEntityBackend& operator=(const MockEntityBackend&) = delete;
	MockEntityBackend& operator=(MockEntityBackend&&) = delete;


};