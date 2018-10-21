#include "Frame.h"

Frame::Frame() :
	id_(0) 
{
	
}

Frame::Frame(const Frame& previous, const Frame& host) :
	id_(previous.id_ + 1)
{

}