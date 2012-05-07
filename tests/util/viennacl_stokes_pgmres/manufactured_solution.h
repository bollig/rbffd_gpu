#ifndef __MANUFACTURED_SOLUTION_H__
#define __MANUFACTURED_SOLUTION_H__

#include "utils/mathematica_base.h"

class ManufacturedSolution : public MathematicaBase
{
    public: 
        virtual double U(double Xx, double Yy, double Zz) {
            double r = (Sqrt(7/Pi)*((524288*Sqrt(15)*Yy*(Power(Xx,2) - Power(Yy,2) + 2*Power(Zz,2)))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),1.5) + 
                        15*Sqrt(286)*((3072*Yy*Power(Power(Xx,2) + Power(Yy,2),1.5)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
                                (3*Power(Power(Xx,2) + Power(Yy,2),3) - 111*Power(Power(Xx,2) + Power(Yy,2),2)*Power(Zz,2) + 364*(Power(Xx,2) + Power(Yy,2))*Power(Zz,4) - 168*Power(Zz,6))*
                                Cos(5*ArcTan(Xx,Yy)))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),4.5) - 
                            (Sqrt(156835045)*Yy*Power(Power(Xx,2) + Power(Yy,2),9)*Zz*Cos(20*ArcTan(Xx,Yy)))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),10) + 
                            Xx*Zz*((3072*Power(Power(Xx,2) + Power(Yy,2),1.5)*Zz*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
                                    (15*Power(Power(Xx,2) + Power(Yy,2),2) - 140*(Power(Xx,2) + Power(Yy,2))*Power(Zz,2) + 168*Power(Zz,4))*Sin(5*ArcTan(Xx,Yy)))/
                                Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),4.5) + (Sqrt(156835045)*Power(Power(Xx,2) + Power(Yy,2),9)*Sin(20*ArcTan(Xx,Yy)))/
                                Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),10)))))/262144.;
            // isNaN
            if (r != r)  {
                return 0.;
            }
            return r; 
        }

        virtual double V(double Xx, double Yy, double Zz) {
            double r = (Sqrt(7/Pi)*((-524288*Sqrt(15)*Xx*(Power(Xx,2) - Power(Yy,2) - 2*Power(Zz,2)))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),1.5) + 
                        15*Sqrt(286)*((-3072*Xx*Power(Power(Xx,2) + Power(Yy,2),1.5)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
                                (3*Power(Power(Xx,2) + Power(Yy,2),3) - 111*Power(Power(Xx,2) + Power(Yy,2),2)*Power(Zz,2) + 364*(Power(Xx,2) + Power(Yy,2))*Power(Zz,4) - 168*Power(Zz,6))*
                                Cos(5*ArcTan(Xx,Yy)))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),4.5) + 
                            (Sqrt(156835045)*Xx*Power(Power(Xx,2) + Power(Yy,2),9)*Zz*Cos(20*ArcTan(Xx,Yy)))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),10) + 
                            Yy*Zz*((3072*Power(Power(Xx,2) + Power(Yy,2),1.5)*Zz*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
                                    (15*Power(Power(Xx,2) + Power(Yy,2),2) - 140*(Power(Xx,2) + Power(Yy,2))*Power(Zz,2) + 168*Power(Zz,4))*Sin(5*ArcTan(Xx,Yy)))/
                                Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),4.5) + (Sqrt(156835045)*Power(Power(Xx,2) + Power(Yy,2),9)*Sin(20*ArcTan(Xx,Yy)))/
                                Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),10)))))/262144.;
            // isNaN
            if (r != r)  {
                return 0.;
            }
            return r; 
        }

        virtual double W(double Xx, double Yy, double Zz) {
            double r = -(Sqrt(7/Pi)*((46080*Sqrt(286)*Power(Power(Xx,2) + Power(Yy,2),2.5)*Zz*
                            (15*Power(Power(Xx,2) + Power(Yy,2),2) - 140*(Power(Xx,2) + Power(Yy,2))*Power(Zz,2) + 168*Power(Zz,4))*Sin(5*ArcTan(Xx,Yy)))/
                        Power(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)),5.5) + Sqrt(5)*
                        (2097152*Sqrt(3)*Xx*Yy*Zz*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),9) + 
                         15*Sqrt(8970964574)*Power(Power(Xx,2) + Power(Yy,2),10)*Sqrt(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))*Sin(20*ArcTan(Xx,Yy)))))/
                (262144.*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),10.5));
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
            double r = (3*Sqrt(7/Pi)*((1267200*Sqrt(286)*Power(Xx,9)*Yy*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5) - 
                        (10137600*Sqrt(286)*Power(Xx,7)*Power(Yy,3)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5) - 
                        (17740800*Sqrt(286)*Power(Xx,5)*Power(Yy,5)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5) + 
                        (6336000*Sqrt(286)*Xx*Power(Yy,9)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5) - 
                        (15206400*Sqrt(286)*Power(Xx,7)*Yy*Power(Zz,2)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5) + 
                        (390297600*Sqrt(286)*Power(Xx,5)*Power(Yy,3)*Power(Zz,2)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5) + 
                        (177408000*Sqrt(286)*Power(Xx,3)*Power(Yy,5)*Power(Zz,2)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5) - 
                        (228096000*Sqrt(286)*Xx*Power(Yy,7)*Power(Zz,2)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5) - 
                        (141926400*Sqrt(286)*Power(Xx,5)*Yy*Power(Zz,4)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5) - 
                        (946176000*Sqrt(286)*Power(Xx,3)*Power(Yy,3)*Power(Zz,4)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5) + 
                        (709632000*Sqrt(286)*Xx*Power(Yy,5)*Power(Zz,4)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5) + 
                        (283852800*Sqrt(286)*Power(Xx,3)*Yy*Power(Zz,6)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5) - 
                        (283852800*Sqrt(286)*Xx*Power(Yy,3)*Power(Zz,6)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5) + 
                        (524288*Sqrt(15)*Power(Xx,2)*Yy)/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),2.5) - 
                        (524288*Sqrt(15)*Power(Yy,3))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),2.5) + 
                        (1048576*Sqrt(15)*Yy*Power(Zz,2))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),2.5) + 
                        (Sqrt(13)*(9975*Sqrt(3450370990)*Power(Xx,18)*Yy*Zz - 508725*Sqrt(3450370990)*Power(Xx,16)*Power(Yy,3)*Zz + 6104700*Sqrt(3450370990)*Power(Xx,14)*Power(Yy,5)*Zz - 
                                   26453700*Sqrt(3450370990)*Power(Xx,12)*Power(Yy,7)*Zz + 48498450*Sqrt(3450370990)*Power(Xx,10)*Power(Yy,9)*Zz - 
                                   39680550*Sqrt(3450370990)*Power(Xx,8)*Power(Yy,11)*Zz + 14244300*Sqrt(3450370990)*Power(Xx,6)*Power(Yy,13)*Zz - 
                                   2034900*Sqrt(3450370990)*Power(Xx,4)*Power(Yy,15)*Zz + 89775*Sqrt(3450370990)*Power(Xx,2)*Power(Yy,17)*Zz - 525*Sqrt(3450370990)*Power(Yy,19)*Zz - 
                                   4096*Power(Xx,19)*(8*Power(Yy,2) + 13*Power(Zz,2)) + 4096*Xx*Power(Yy,2)*Power(Power(Yy,2) + Power(Zz,2),7)*
                                   (8*Power(Yy,4) - 85*Power(Yy,2)*Power(Zz,2) - 60*Power(Zz,4)) - 4096*Power(Xx,17)*(56*Power(Yy,4) - 3*Power(Yy,2)*Power(Zz,2) + 71*Power(Zz,4)) - 
                                   28672*Power(Xx,11)*Power(Power(Yy,2) + Power(Zz,2),2)*(16*Power(Yy,6) - 350*Power(Yy,4)*Power(Zz,2) - 500*Power(Yy,2)*Power(Zz,4) - 35*Power(Zz,6)) - 
                                   28672*Power(Xx,13)*(Power(Yy,2) + Power(Zz,2))*(32*Power(Yy,6) - 220*Power(Yy,4)*Power(Zz,2) - 280*Power(Yy,2)*Power(Zz,4) + 5*Power(Zz,6)) + 
                                   4096*Power(Xx,3)*Power(Power(Yy,2) + Power(Zz,2),6)*(56*Power(Yy,6) - 445*Power(Yy,4)*Power(Zz,2) - 250*Power(Yy,2)*Power(Zz,4) + 20*Power(Zz,6)) + 
                                   28672*Power(Xx,7)*Power(Power(Yy,2) + Power(Zz,2),4)*(32*Power(Yy,6) - 4*Power(Yy,4)*Power(Zz,2) + 176*Power(Yy,2)*Power(Zz,4) + 47*Power(Zz,6)) + 
                                   28672*Power(Xx,9)*Power(Power(Yy,2) + Power(Zz,2),3)*(16*Power(Yy,6) + 238*Power(Yy,4)*Power(Zz,2) + 448*Power(Yy,2)*Power(Zz,4) + 61*Power(Zz,6)) + 
                                   4096*Power(Xx,5)*Power(Power(Yy,2) + Power(Zz,2),5)*(160*Power(Yy,6) - 764*Power(Yy,4)*Power(Zz,2) - 104*Power(Yy,2)*Power(Zz,4) + 127*Power(Zz,6)) - 
                                   4096*Power(Xx,15)*(160*Power(Yy,6) - 356*Power(Yy,4)*Power(Zz,2) - 416*Power(Yy,2)*Power(Zz,4) + 133*Power(Zz,6))))/
                        Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),11)))/65536.;
            // isNaN
            if (r != r)  {
                return 0.;
            }
            return r; 
        }

        virtual double RHS_V(double Xx, double Yy, double Zz) {
            double r = (3*Sqrt(7/Pi)*((137625600*Sqrt(44854822870)*Power(Xx,19)*Zz)/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),11) - 
                        (653721600*Sqrt(44854822870)*Power(Xx,17)*(Power(Xx,2) + Power(Yy,2))*Zz)/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),11) + 
                        (1307443200*Sqrt(44854822870)*Power(Xx,15)*Power(Power(Xx,2) + Power(Yy,2),2)*Zz)/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),11) - 
                        (1430016000*Sqrt(44854822870)*Power(Xx,13)*Power(Power(Xx,2) + Power(Yy,2),3)*Zz)/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),11) + 
                        (929510400*Sqrt(44854822870)*Power(Xx,11)*Power(Power(Xx,2) + Power(Yy,2),4)*Zz)/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),11) - 
                        (365164800*Sqrt(44854822870)*Power(Xx,9)*Power(Power(Xx,2) + Power(Yy,2),5)*Zz)/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),11) + 
                        (84268800*Sqrt(44854822870)*Power(Xx,7)*Power(Power(Xx,2) + Power(Yy,2),6)*Zz)/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),11) - 
                        (10533600*Sqrt(44854822870)*Power(Xx,5)*Power(Power(Xx,2) + Power(Yy,2),7)*Zz)/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),11) - 
                        (6758400*Sqrt(286)*Power(Xx,6)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
                         (3*Power(Power(Xx,2) + Power(Yy,2),2) - 96*(Power(Xx,2) + Power(Yy,2))*Power(Zz,2) + 224*Power(Zz,4)))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5) + 
                        4*Sqrt(5)*Power(Xx,3)*((149625*Sqrt(8970964574)*Power(Power(Xx,2) + Power(Yy,2),8)*Zz)/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),11) - 
                            (262144*Sqrt(3))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),2.5)) + 
                        Sqrt(5)*Xx*((-9975*Sqrt(8970964574)*Power(Power(Xx,2) + Power(Yy,2),9)*Zz)/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),11) - 
                            (524288*Sqrt(3)*Power(Xx,2))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),2.5) - 
                            (524288*Sqrt(3)*Power(Yy,2))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),2.5) + (1048576*Sqrt(3))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),1.5)) + 
                        2048*Sqrt(13)*Power(Xx,4)*((-799425*Sqrt(22)*Power(Zz,6)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5) + 
                            (1472625*Sqrt(22)*Power(Zz,4)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),4.5) + 
                            (32*Yy)/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),3) + (12375*Sqrt(22)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/
                            Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),2.5) + Power(Zz,2)*
                            ((-528*Yy)/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),4) - 
                             (408375*Sqrt(22)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),3.5))) + 
                        512*Sqrt(13)*Power(Zz,2)*((266475*Sqrt(22)*Power(Zz,8)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5) - 
                                (673200*Sqrt(22)*Power(Zz,6)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),4.5) - 
                                (104*Yy)/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),2) + 
                                (12375*Sqrt(22)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),1.5) + 
                                66*Power(Zz,4)*((-4*Yy)/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),4) + 
                                    (8475*Sqrt(22)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),3.5)) + 
                                8*Power(Zz,2)*((46*Yy)/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),3) - 
                                    (20625*Sqrt(22)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),2.5))) + 
                        512*Sqrt(13)*Power(Xx,2)*((799425*Sqrt(22)*Power(Zz,8)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5) - 
                                (168300*Sqrt(22)*Power(Zz,6)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),4.5) - 
                                (64*Yy)/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),2) - (12375*Sqrt(22)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/
                                Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),1.5) + 66*Power(Zz,4)*
                                ((-32*Yy)/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),4) - 
                                 (14625*Sqrt(22)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),3.5)) + 
                                Power(Zz,2)*((1536*Yy)/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),3) + 
                                    (346500*Sqrt(22)*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2))))/Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),2.5)))))/65536.;
            // isNaN
            if (r != r)  {
                return 0.;
            }
            return r; 
        }

        virtual double RHS_W(double Xx, double Yy, double Zz) {
            double r = (3*Sqrt(7/Pi)*(4096*Sqrt(13)*Power(Power(Xx,2) + Power(Yy,2),2.5)*Zz*(13*(Power(Xx,2) + Power(Yy,2)) - 20*Power(Zz,2))*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),7)*
                        Cos(4*ArcTan(Xx,Yy)) - 422400*Sqrt(286)*Power(Power(Xx,2) + Power(Yy,2),3)*Zz*Sqrt(1/(Power(Xx,2) + Power(Yy,2) + Power(Zz,2)))*
                        Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),5.5)*(15*Power(Power(Xx,2) + Power(Yy,2),2) - 140*(Power(Xx,2) + Power(Yy,2))*Power(Zz,2) + 168*Power(Zz,4))*
                        Sin(5*ArcTan(Xx,Yy)) - Sqrt(5)*Sqrt(Power(Xx,2) + Power(Yy,2))*
                        (2097152*Sqrt(3)*Xx*Yy*Zz*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),8.5) + 525*Sqrt(8970964574)*Power(Power(Xx,2) + Power(Yy,2),10)*Sin(20*ArcTan(Xx,Yy)))))/
                (65536.*Sqrt(Power(Xx,2) + Power(Yy,2))*Power(Power(Xx,2) + Power(Yy,2) + Power(Zz,2),11));
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
