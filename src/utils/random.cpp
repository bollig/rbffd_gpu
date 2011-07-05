#include "random.h"

//----------------------------------------------------------------------

int randi(int i1, int i2) {
	double r = (double) (random() / RAND_MAX);
	return (int)(i1 + r * (i2 - i1));
}

//----------------------------------------------------------------------

double randf(double f1, double f2) {
	return f1 + (f2 - f1)*(random() / (double)RAND_MAX);
}


//----------------------------------------------------------------------

#if 0
double random(double a, double b)
{
        // use system version of random, not class version
        double r = ::random() / (double) RAND_MAX;
        return a + r*(b-a);
}
#endif 
//----------------------------------------------------------------------
