#include "Character.h"

void Character::ResetCallbackInfo(){
	for (int i = 0; i < CallbackInfo.size(); i++){
		(*CallbackInfo[i])=false;
	}
}