#ifndef _CLOCK_H_
#define _CLOCK_H_

// clock measures processor time

#include <string>
#include "time.h"

class Clock
{
private:
	float t;
	clock_t t1;
	clock_t t2;
	float scale;
	std::string name;
	std::string unit;
	int count;

public:
	Clock(const char* name);
	~Clock();
	void reset();
	void begin();
	void end();

	void stop() { end(); }
	void start() { begin(); }
	void print();
};

#endif
