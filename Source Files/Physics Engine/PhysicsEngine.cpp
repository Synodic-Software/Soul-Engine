//---------------------------------------------------------------------------------------------------
//@file	N:\Documents\Soul Engine\Source Files\Physics Engine\PhysicsEngine.cpp.
//Implements the physics engine class.

#include "PhysicsEngine.h"
#include "CUDA/PhysicsEngine.cuh"

//---------------------------------------------------------------------------------------------------
//this just processes a scene with the CUDA physics engine.
//@param	scene	The scene.

void PhysicsEngine::Process(const Scene* scene){

	ProcessScene(scene);
}
