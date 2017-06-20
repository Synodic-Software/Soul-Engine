//---------------------------------------------------------------------------------------------------
//@file	N:\Documents\Soul Engine\Source Files\Engine Core\Object\Character\Character.cpp.
//Implements the character class.

#include "Character.h"

//Resets the callback information.
void Character::ResetCallbackInfo(){
	for (int i = 0; i < CallbackInfo.size(); i++){
		(*CallbackInfo[i])=false;
	}
}