#include "Hand.h"


Hand::Hand()
{
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
	ExtractFromFile("rebellion.obj");
}