#include <sys/time.h>
#include "timege.h"

using namespace GE;

//----------------------------------------------------------------------
Time::Time(const char* name_)
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
		printf("Time does handle this case\n");
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
Time::~Time()
{
}
//----------------------------------------------------------------------
void Time::reset()
{
	t = 0.0;
	t1 = clock();
	count = 0;

    //getrusage(RUSAGECHILDREN, &ru);
    //tmp[0] = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6;
    //tmp[1] = ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
  //struct rusage ru;
  //float tmp[2];
  //enum {CHILDREN=0, SELF=1};
  //enum {RUSAGESELF=0, RUSAGECHILDREN=-1};


}
//----------------------------------------------------------------------
void Time::begin()
{
	gettimeofday(&t_start, NULL);
	t1 = clock();
	t2 = 0.0;
	count++;
}
//----------------------------------------------------------------------
void Time::end()
{
	gettimeofday(&t_end, NULL);
	double tt = (t_end.tv_sec - t_start.tv_sec) +
	     (t_end.tv_usec - t_start.tv_usec) * 1.e-6;
	//printf("tt= %f\n", tt);
	t += 1000*tt;

	//t +=  (clock() - t1) * scale;
}
//----------------------------------------------------------------------
void Time::print()
{
	printf("%s: tot: %f, avg: %f (ms), (count=%d)\n", name.c_str(), t, t/count, count);
}
//----------------------------------------------------------------------
