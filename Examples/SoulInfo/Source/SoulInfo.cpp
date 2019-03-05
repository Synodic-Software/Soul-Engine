#include "SoulInfo.h"

SoulInfo::SoulInfo(SoulParameters params) :
	SoulApplication(params) {

}

int main(int, char*[])
{
	//app params
	SoulParameters appParams;
	SoulInfo app(appParams);

	app.Run();

}