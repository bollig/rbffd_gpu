#ifndef _RANDOM_HPP_
#define _RANDOM_HPP_

#include <time.h>
#include <stdlib.h>

inline void init()
{
	static bool init = false;
	if (!init)
	{
		srand( (unsigned int)time(NULL) );
		init = true;
	}
}

template<class TYPE>
TYPE random();

template<>
double random<double>()
{
  init();
  return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

template<>
float random<float>()
{
  init();
  return static_cast<float>(random<double>());
}

#endif

