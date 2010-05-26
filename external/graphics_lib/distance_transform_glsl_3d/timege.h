#ifndef _TIMEGE_H_
#define _TIMEGE_H_

// gettimeofday: measured in sec/microsec: wall clock time
// irrespective of CPU/system/threads, etc.

#include <string>
#include <sys/time.h>
//#include "time.h"

namespace GE {

class Time
{
private:
	struct timeval t_start, t_end;
	double elapsed;
	float t;
	clock_t t1;
	clock_t t2;
	float scale;
	std::string name;
	std::string unit;
	int count;

public:
	Time(const char* name);
	~Time();
	void reset();
	void begin();
	void end();

	void stop() { end(); }
	void start() { begin(); }
	void print();
};

}

#endif
