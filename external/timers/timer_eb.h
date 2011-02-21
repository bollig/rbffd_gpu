#ifndef _TIME_EB_H_
#define _TIME_EB_H_

// gettimeofday: measured in sec/microsec: wall clock time
// irrespective of CPU/system/threads, etc.

#include <string>
#include <sys/time.h>
#include <vector>
#include <stdio.h>
#include <map> 
#include <string> 
//#include "time.h"

namespace EB {

class Timer
{
public:
	static std::vector<Timer*> timeList;

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
	int nbCalls;
	int offset;

public:
	// nbCalls: how many calls before resetting the clock
	// if nbCalls not -1, print time after nbCalls calls
	// offset: how many calls to ignore
	Timer();
	Timer(const char* name, int offset=0, int nbCalls=-1);
	Timer(const Timer&);
	~Timer();
	void reset();
	void begin();
	void end();
	int getCount() { return count;}

	void stop() { end(); }
	void start() { begin(); }
	static void printAll(FILE* fd=stdout, int label_width=50);
	void print(FILE* fd=stdout, int label_width=50);
    void writeAllToFile(std::string filename="timer_log"); 
	void printReset();
};



class TimerList : public std::map<std::string, EB::Timer*>
{
    public: 
    void writeToFile(std::string filename) {
        (*(this->begin())).second->writeAllToFile(filename); 
    } 
    void printAll() {
        (*(this->begin())).second->printAll(); 
    } 
};

};
#endif
