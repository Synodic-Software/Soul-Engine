#include "Hand.h"
//deal with hand model in soul engine

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
	ExtractFromFile("lost_empire.obj");
}