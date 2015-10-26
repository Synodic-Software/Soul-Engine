#include "ObjectSceneAbstraction.cuh"


ObjectSceneAbstraction::ObjectSceneAbstraction(Object* obj)
{
	object = obj;
	nextObject = NULL;
}


ObjectSceneAbstraction::~ObjectSceneAbstraction()
{

	if (nextObject!=NULL){
		delete nextObject;
	}

}
