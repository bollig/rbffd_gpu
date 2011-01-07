#include <stdlib.h>

int randi(int i1, int i2) {
	double r = (double) (random() / RAND_MAX);
	return (i1 + r * (i2 - i1));
}


double randf(double f1, double f2) {
	return f1 + (f2 - f1)*(random() / RAND_MAX);
}

