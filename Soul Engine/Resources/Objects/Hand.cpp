#include "Hand.h"
//deal with hand model in soul engine
/*
this is now deprecated
from a point early in dev
objects were originally hardcoded in 
this is now *probably* unneeded
gonna leave it just in case
*/

//constructor for hand
Hand::Hand(glm::vec3 it)
{
	xyzPosition = it;
	Load();
}

//destructor for hand
Hand::~Hand()
{
}

//update position
void Hand::Update(double){

}

//update position late
void Hand::UpdateLate(double){

}

//load a model for the hand
void Hand::Load(){
	ExtractFromFile("lost_empire.obj");
}