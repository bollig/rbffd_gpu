#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "timer_eb.h"

using namespace EB;

// Modified Gordons timer and ran into compilation and linking 
// problems so Ive been forced to rename to my own timer


// Must initialize in cpp file to avoid multiple definitions
std::vector<EB::Timer*> EB::Timer::timeList;

//----------------------------------------------------------------------
Timer::Timer()
{
	static int const_count = 0; 
	if (!const_count) {
		timeList.resize(0); 
	}
	name = "";
	scale = 0.;
	count = 0;
	unit = "ms";
	t = 0.0; 
	t1 = 0;
	t2 = 0;

	this->nbCalls = 0;
	this->offset = 0;
	reset();
}
//----------------------------------------------------------------------
Timer::Timer(const char* name_, int offset, int nbCalls)
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
		printf("Timer does handle this case\n");
		printf("CLOCKS_PER_SEC= %ld\n", (long) CLOCKS_PER_SEC);
		exit(0);
	}
	count = 0;
	unit = "ms";
	t = 0.0;
	t1 = 0;
	t2 = 0;

	this->nbCalls = nbCalls;
	this->offset = offset;
	timeList.push_back(this);
	//printf("constructor: this= %d, name= %s\n", this, name.c_str());
	reset();
}
//----------------------------------------------------------------------
Timer::Timer(const Timer& t)
{
	name = t.name;
	scale = t.scale;
	count = t.count;
	this->t = t.t;
	this->t1 = t.t1;
	this->t2 = t.t2;
	this->nbCalls = t.nbCalls;
	this->offset = t.offset;
	timeList.push_back(this);
	reset();
}
//----------------------------------------------------------------------
Timer::~Timer()
{
}
//----------------------------------------------------------------------
void Timer::reset()
{
	t = 0.0;
	t1 = clock();
	count = 0;
}
//----------------------------------------------------------------------
void Timer::begin()
{
	if (count < offset) {
		count++;
		return;
	}
	gettimeofday(&t_start, NULL);
	t1 = clock();
	t2 = 0.0;
	count++;
}
//----------------------------------------------------------------------
void Timer::end()
{
	if (count <= offset) return;

	gettimeofday(&t_end, NULL);
	double tt = (t_end.tv_sec - t_start.tv_sec) +
	     (t_end.tv_usec - t_start.tv_usec) * 1.e-6;
	//printf("tt= %f\n", tt);
	t += 1000*tt;

	if (count == nbCalls) {
		print();
		reset();
	}

	//t +=  (clock() - t1) * scale;
}
//----------------------------------------------------------------------
void Timer::print()
{
	if (count <= 0) return;
	int real_count = count - offset;
	printf("%s:\t avg: %g,\t tot: %g\t (count=%d)\n", 
		name.c_str(), t/real_count, t, real_count);
}
//----------------------------------------------------------------------
void Timer::printReset()
{
	//end();
	// I would rather control end() myself
	print();
	reset();
}
//----------------------------------------------------------------------
void Timer::printAll()
{
#if 1
	printf("====================================\n"); 
	printf("Timers [All times in ms (1/1000 s)]: \n"); 		
	printf("====================================\n\n"); 

	for (int i=0; i < timeList.size(); i++) {
		Timer& tim = *(timeList[i]);
		tim.print();
	}
	printf("\nNOTE: only timers that have called Timer::start() are shown. \n");
	printf("      [A time of 0.0 may indicate the timer was not stopped.]\n"); 
	printf("====================================\n"); 
#endif
}
//----------------------------------------------------------------------
