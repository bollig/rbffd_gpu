#include "clock.h"

//----------------------------------------------------------------------
Clock::Clock(const char* name_)
{
	name = name_;

	switch (CLOCKS_PER_SEC) {
	case 1000000:
		scale = 1000. / (float) CLOCKS_PER_SEC;
		break;
	case 1000:
		scale = 1. / (float) CLOCKS_PER_SEC;
		break;
	default:
		printf("Clock does handle this case\n");
		printf("CLOCKS_PER_SEC= %d\n", CLOCKS_PER_SEC);
		exit(0);
	}
	count = 0;
	unit = "ms";
	t = 0.0;
	t1 = 0;
	t2 = 0;
	reset();
}
//----------------------------------------------------------------------
Clock::~Clock()
{
}
//----------------------------------------------------------------------
void Clock::reset()
{
	t = 0.0;
	t1 = clock();
	count = 0;
}
//----------------------------------------------------------------------
void Clock::begin()
{
	t1 = clock();
	t2 = 0.0;
	count++;
}
//----------------------------------------------------------------------
void Clock::end()
{
	t +=  (clock() - t1) * scale;
}
//----------------------------------------------------------------------
void Clock::print()
{
	printf("%s: %f (%s), (count=%d)\n", name.c_str(), t/count, unit.c_str(), count);
}
//----------------------------------------------------------------------
