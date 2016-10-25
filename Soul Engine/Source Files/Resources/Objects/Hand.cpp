#include "Hand.h"
//deal with hand model in soul engine
/*
this is now deprecated
from a point early in dev
objects were originally hardcoded in 
this is now *probably* unneeded
gonna leave it just in case
*/
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