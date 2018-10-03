#pragma once

class AbstractNode{

public:

	AbstractNode() = default;
	virtual ~AbstractNode() = default;

	AbstractNode(const AbstractNode&) = delete;
	AbstractNode(AbstractNode&& o) noexcept = delete;

	AbstractNode& operator=(const AbstractNode&) = delete;
	AbstractNode& operator=(AbstractNode&& other) noexcept = delete;

};