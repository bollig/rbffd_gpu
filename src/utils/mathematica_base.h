#ifndef __MATHEMATICA_BASE_H__
#define __MATHEMATICA_BASE_H__

#include <stdlib.h>
#include <math.h> 

#define Pi M_PI

class MathematicaBase
{
    public:
        /***********   MATHEMATICA mdefs.h PROVIDED MOST OF THESE ************/
        // Its ok to directly use their definitions

        // NOTE: I found that RHEL6 (not sure if its just this version) was
        // MUCH slower when using the double precision version of
        // pow(double,double) vs pow(float,float). For spherical harmonics,
        // we do not see a noticable difference in the harmonic
        // approximations caused by downcasting to float here. However, the
        // impact on performance is 1000x speedup for 10201 nodes. This
        // casting causes 1e-5 error in production of sph32105, no noticeable
        // error in sph32 or sph2020.  Presumably the lack of error is
        // because the pow is integer, not double/float.  
#if 0
        double Power(double x, double y) { return    (pow((double)(x), (double)(y))) ; }
#else 
        inline double Power(double x, double y) { return    (double)(pow((float)(x), (float)(y))) ; }
#endif 
        inline double Sqrt(double x)  { return (sqrt((double)(x))) ; }

        inline double Abs(double x)   { return (fabs((double)(x))) ; }

        inline double Exp(double x)   { return (exp((double)(x))) ; } 
        inline double Log(double x)   { return (log((double)(x))) ; }

        inline double Sin(double x)   { return (sin((double)(x))) ; }
        inline double Cos(double x)   { return (cos((double)(x))) ; }
        inline double Tan(double x)   { return (tan((double)(x))) ; }

        inline double ArcSin(double x){ return (asin((double)(x))) ; }
        inline double ArcCos(double x){ return (acos((double)(x))) ; }
        inline double ArcTan(double x){ return (atan((double)(x))) ; }
        inline double ArcTan(double x,double y){ return (atan2((double)(y), (double)(x))) ; }

        inline double Sinh(double x)  { return (sinh((double)(x))) ; }
        inline double Cosh(double x)  { return (cosh((double)(x))) ; }
        inline double Tanh(double x)  { return (tanh((double)(x))) ; }

        // I Know M_PI is defined in math.h 
        //        constexpr static double Pi = M_PI;
        /***********   END MATHEMATICA mdefs.h PROVIDED ************/
};


#endif 
