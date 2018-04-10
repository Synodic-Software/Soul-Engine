/*
	Please refer to this site for all OS macros
	https://sourceforge.net/p/predef/wiki/OperatingSystems/
*/

#ifdef __CYGWIN__
#include "cygwin.h"
#endif

#ifdef _WIN32
#include "windows32.h"
#endif

#ifdef _WIN64
#include "windows64.h"
#endif

#ifdef 	__GNU__
#include "ghu.h"
#endif

#ifdef 	__MACH__
#include "osx.h"
#endif

#ifdef 	__APPLE__
#include "osx.h"
#endif

int main(){
	printOS();
	return 0;
}