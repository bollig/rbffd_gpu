#include <stdio.h>
#include <math.h>
#include <time.h>

int main()
{
	clock_t t1 = clock();

	float a = 1.;
	for (int i=0; i < 1000; i++) {
		a = sin(a);
	}
	printf("a = %f\n", a);

	clock_t t2 = clock() - t1;

	printf("t1= %d, t2= %d\n", t1, t2);
	printf("time: %f\n", t2 / CLOCKS_PER_SEC);
	printf("clocks per sec: %d\n", CLOCKS_PER_SEC);

	if (CLOCKS_PER_SEC == 1000000) {
		printf("time in microsec\n");
	} else if (CLOCKS_PER_SEC == 1000) {
		printf("time in microsec\n");
	}
}
