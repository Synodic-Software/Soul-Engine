#include <iostream>

#ifdef CYGWIN_H
void printOS(){
	std::cout<<"OS: cygwin"<<std::endl;
}
#endif