#ifndef __MANUFACTURED_SOLUTION_ALT_H__
#define __MANUFACTURED_SOLUTION_ALT_H__

#include "utils/mathematica_base.h"

// This is just the g(x) = 8*Y_3^2 - 3*Y_10^5

class ManufacturedSolutionAlt : public MathematicaBase
{
    public: 
        virtual double U(double Xx, double Yy, double Zz) {
            double r = (Sqrt(7/Pi)*(512*Sqrt(15)*Yy*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),3)*(Power(Xx,2) - Power(Yy,2) + 2*Power(Zz,2)) + 
       45*Sqrt(286)*Power(Power(Xx,2) + Power(Yy,2),1.5)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
        (Yy*(3*Power(Power(Xx,2) + Power(Yy,2),3) - 111*Power(Power(Xx,2) + Power(Yy,2),2)*Power(Zz,2) + 364*(Power(Xx,2) + Power(Yy,2))*Power(Zz,4) - 168*Power(Zz,6))*Cos(5*ArcTan(Xx,Yy)) + 
          Xx*Power(Zz,2)*(15*Power(Power(Xx,2) + Power(Yy,2),2) - 140*(Power(Xx,2) + Power(Yy,2))*Power(Zz,2) + 168*Power(Zz,4))*Sin(5*ArcTan(Xx,Yy)))))/
   (256.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),4.5)); 

            // isNaN
            if (r != r)  {
                return 0.;
            }
            return r; 
        }

        virtual double V(double Xx, double Yy, double Zz) {
            double r = (Sqrt(7/Pi)*(-512*Sqrt(15)*Xx*(Power(Xx,2) - Power(Yy,2) - 2*Power(Zz,2))*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),3) - 
       45*Sqrt(286)*Power(Power(Xx,2) + Power(Yy,2),1.5)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
        (Xx*(3*Power(Power(Xx,2) + Power(Yy,2),3) - 111*Power(Power(Xx,2) + Power(Yy,2),2)*Power(Zz,2) + 364*(Power(Xx,2) + Power(Yy,2))*Power(Zz,4) - 168*Power(Zz,6))*Cos(5*ArcTan(Xx,Yy)) - 
          Yy*Power(Zz,2)*(15*Power(Power(Xx,2) + Power(Yy,2),2) - 140*(Power(Xx,2) + Power(Yy,2))*Power(Zz,2) + 168*Power(Zz,4))*Sin(5*ArcTan(Xx,Yy)))))/
   (256.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),4.5)); 

            // isNaN
            if (r != r)  {
                return 0.;
            }
            return r; 
        }

        virtual double W(double Xx, double Yy, double Zz) {
            double r = -(Sqrt(7/Pi)*Zz*(2048*Sqrt(15)*Xx*Yy*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),3) + 
        45*Sqrt(286)*Power(Power(Xx,2) + Power(Yy,2),2.5)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
         (15*Power(Power(Xx,2) + Power(Yy,2),2) - 140*(Power(Xx,2) + Power(Yy,2))*Power(Zz,2) + 168*Power(Zz,4))*Sin(5*ArcTan(Xx,Yy))))/(256.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),4.5)); 

            // isNaN
            if (r != r)  {
                return 0.;
            }
            return r; 
        }

        virtual double P(double Xx, double Yy, double Zz) {
            double r = (-3*Sqrt(91/Pi)*Power(Power(Xx,2) + Power(Yy,2),2)*(Power(Xx,2) + Power(Yy,2) - 10*Power(Zz,2))*Cos(4*ArcTan(Xx,Yy)))/(32.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),3));

            // isNaN
            if (r != r)  {
                return 0.;
            }
            return r; 
        }

        virtual double RHS_U(double Xx, double Yy, double Zz) {
            double r = (3*Sqrt(7/Pi)*(1024*Sqrt(15)*Power(Xx,8)*Yy + 9216*Sqrt(15)*Power(Xx,4)*Yy*Power(Zz,2)*(Power(Yy,2) + Power(Zz,2)) - 
       1024*Sqrt(15)*Power(Xx,2)*Yy*(2*Power(Yy,2) - 7*Power(Zz,2))*Power(Power(Yy,2) + Power(Zz,2),2) - 1024*Sqrt(15)*Yy*(Power(Yy,2) - 2*Power(Zz,2))*Power(Power(Yy,2) + Power(Zz,2),3) + 
       2475*Sqrt(286)*Power(Xx,9)*Yy*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) + 1024*Sqrt(15)*Power(Xx,6)*Yy*(2*Power(Yy,2) + 5*Power(Zz,2)) - 
       4*Sqrt(13)*Power(Xx,7)*(4950*Sqrt(22)*Power(Yy,3)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) + 7425*Sqrt(22)*Yy*Power(Zz,2)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) + 
          16*Power(Yy,2)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 26*Power(Zz,2)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) + 
       2*Sqrt(13)*Power(Xx,5)*(-17325*Sqrt(22)*Power(Yy,5)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) + 381150*Sqrt(22)*Power(Yy,3)*Power(Zz,2)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) - 
          138600*Sqrt(22)*Yy*Power(Zz,4)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) - 32*Power(Yy,4)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 
          516*Power(Yy,2)*Power(Zz,2)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 28*Power(Zz,4)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) + 
       Sqrt(13)*Xx*Power(Yy,2)*(12375*Sqrt(22)*Power(Yy,7)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) - 445500*Sqrt(22)*Power(Yy,5)*Power(Zz,2)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) + 
          1386000*Sqrt(22)*Power(Yy,3)*Power(Zz,4)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) - 554400*Sqrt(22)*Yy*Power(Zz,6)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) + 
          64*Power(Yy,6)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) - 616*Power(Yy,4)*Power(Zz,2)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) - 
          1160*Power(Yy,2)*Power(Zz,4)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) - 480*Power(Zz,6)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) + 
       4*Sqrt(13)*Power(Xx,3)*(86625*Sqrt(22)*Power(Yy,5)*Power(Zz,2)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) - 
          462000*Sqrt(22)*Power(Yy,3)*Power(Zz,4)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) + 138600*Sqrt(22)*Yy*Power(Zz,6)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) + 
          16*Power(Yy,6)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 130*Power(Yy,4)*Power(Zz,2)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 
          220*Power(Yy,2)*Power(Zz,4)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 40*Power(Zz,6)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))))/
   (128.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5)); 

            // isNaN
            if (r != r)  {
                return 0.;
            }
            return r; 
        }

        virtual double RHS_V(double Xx, double Yy, double Zz) {
            double r = (3*Sqrt(7/Pi)*(-1024*Sqrt(15)*Power(Xx,9) + 3072*Sqrt(15)*Power(Xx,5)*Power(Zz,2)*(Power(Yy,2) + Power(Zz,2)) - 2475*Sqrt(286)*Power(Xx,10)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) - 
       1024*Sqrt(15)*Power(Xx,7)*(2*Power(Yy,2) + Power(Zz,2)) + 1024*Sqrt(15)*Xx*Power(Power(Yy,2) + Power(Zz,2),3)*(Power(Yy,2) + 2*Power(Zz,2)) + 
       1024*Sqrt(15)*Power(Xx,3)*Power(Power(Yy,2) + Power(Zz,2),2)*(2*Power(Yy,2) + 5*Power(Zz,2)) + 
       Sqrt(13)*Power(Xx,8)*(19800*Sqrt(22)*Power(Yy,2)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) + 91575*Sqrt(22)*Power(Zz,2)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) + 
          64*Yy*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) + 2*Sqrt(13)*Power(Xx,6)*
        (17325*Sqrt(22)*Power(Yy,4)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) - 381150*Sqrt(22)*Power(Yy,2)*Power(Zz,2)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) - 
          150150*Sqrt(22)*Power(Zz,4)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) + 32*Power(Yy,3)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) - 
          308*Yy*Power(Zz,2)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) + Sqrt(13)*Power(Yy,3)*Power(Zz,2)*
        (12375*Sqrt(22)*Power(Yy,5)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) - 115500*Sqrt(22)*Power(Yy,3)*Power(Zz,2)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) + 
          138600*Sqrt(22)*Yy*Power(Zz,4)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) - 104*Power(Yy,4)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 
          56*Power(Yy,2)*Power(Zz,2)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 160*Power(Zz,4)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) + 
       2*Sqrt(13)*Power(Xx,4)*(-259875*Sqrt(22)*Power(Yy,4)*Power(Zz,2)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) + 
          1212750*Sqrt(22)*Power(Yy,2)*Power(Zz,4)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) + 69300*Sqrt(22)*Power(Zz,6)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) - 
          32*Power(Yy,5)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 260*Power(Yy,3)*Power(Zz,2)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) - 
          580*Yy*Power(Zz,4)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) - Sqrt(13)*Power(Xx,2)*Yy*
        (12375*Sqrt(22)*Power(Yy,7)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) - 346500*Sqrt(22)*Power(Yy,5)*Power(Zz,2)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) + 
          346500*Sqrt(22)*Power(Yy,3)*Power(Zz,4)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) + 831600*Sqrt(22)*Yy*Power(Zz,6)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))) + 
          64*Power(Yy,6)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) - 1032*Power(Yy,4)*Power(Zz,2)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) - 
          880*Power(Yy,2)*Power(Zz,4)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)) + 480*Power(Zz,6)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))))/
   (128.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5)); 

            // isNaN
            if (r != r)  {
                return 0.;
            }
            return r; 
        }

        virtual double RHS_W(double Xx, double Yy, double Zz) {
            double r = (-3*Sqrt(7/Pi)*Zz*(4096*Sqrt(15)*Xx*Yy*Sqrt(Power(Xx,2) + Power(Yy,2))*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),3) - 
       8*Sqrt(13)*Power(Power(Xx,2) + Power(Yy,2),2.5)*(13*(Power(Xx,2) + Power(Yy,2)) - 20*Power(Zz,2))*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),1.5)*Cos(4*ArcTan(Xx,Yy)) + 
       825*Sqrt(286)*Power(Power(Xx,2) + Power(Yy,2),3)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
        (15*Power(Power(Xx,2) + Power(Yy,2),2) - 140*(Power(Xx,2) + Power(Yy,2))*Power(Zz,2) + 168*Power(Zz,4))*Sin(5*ArcTan(Xx,Yy))))/
   (128.*Sqrt(Power(Xx,2) + Power(Yy,2))*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5)); 

            // isNaN
            if (r != r)  {
                return 0.;
            }
            return r; 
        }

        virtual double RHS_P(double Xx, double Yy, double Zz) {
            return 0.; 
        }


        // Approximate integrals computed as sum(U_exact(s1:s2)) where s1 = i*N+1; s = i*N+10201; (so integral over 10201 equally weighted nodes)
#if 1
        virtual double RHS_CU() { return 8.634921215; }
        virtual double RHS_CV() { return -7.204768375; }
        virtual double RHS_CW() { return 3.949225301; }
        virtual double RHS_CP() { return -0.114522888; }
#else 
        virtual double RHS_CU() { return 0.; }
        virtual double RHS_CV() { return 0.; }
        virtual double RHS_CW() { return 0.; }
        virtual double RHS_CP() { return 0.; }
#endif 
};


#endif 
