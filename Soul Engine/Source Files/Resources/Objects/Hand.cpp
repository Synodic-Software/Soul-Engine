#include "Hand.h"


Hand::Hand(glm::vec3 it)
{
	xyzPosition = it;
	Load();
}


Hand::~Hand()
{
}

void Hand::Update(double){

}
void Hand::UpdateLate(double){

}
void Hand::Load(){
	ExtractFromFile("sponza.obj");
}