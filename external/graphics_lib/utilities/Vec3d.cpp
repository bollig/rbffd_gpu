
/*
 * Written by Dr. Gordon Erlebacher.
 */

#include <math.h>
#include <stdio.h>
#include <iostream>
//#include <Amira/HxMessage.h>
#include "Vec3d.h"

using namespace std;

//=======================================================================

#if 0
Vec3d::Vec3d()
{
    Vec3d((double)0., (double)0., (double)0.);
}

//=======================================================================

Vec3d::Vec3d(int x, int y, int z)
{
    vec[0] = (double) x;
    vec[1] = (double) y;
    vec[2] = (double) z;
}

//=======================================================================

Vec3d::Vec3d(float x, float y, float z)
{
    vec[0] = x;
    vec[1] = y;
    vec[2] = z;
}

//=======================================================================

Vec3d::Vec3d(double x, double y, double z)
{
    vec[0] = x;
    vec[1] = y;
    vec[2] = z;
}

//=======================================================================

Vec3d::Vec3d(double* pt)
{
    vec[0] = *pt++;
    vec[1] = *pt++;
    vec[2] = *pt;
}

//=======================================================================

Vec3d::Vec3d(double* pt)
{
    vec[0] = *pt++;
    vec[1] = *pt++;
    vec[2] = *pt;
}
#endif

//=======================================================================

Vec3d::Vec3d(Vec3d& vec)
{
    this->vec[0] = vec.x();
    this->vec[1] = vec.y();
    this->vec[2] = vec.z();
}

//=======================================================================

double* Vec3d::getVec()
{
    return vec;
}

//=======================================================================

void Vec3d::getVec(double* x, double* y, double* z)
{
    *x = vec[0];
    *y = vec[1];
    *z = vec[2];
}

//=======================================================================
void Vec3d::setValue(double x)
{
    vec[0] = x;
    vec[1] = x;
    vec[2] = x;
}
//=======================================================================

void Vec3d::setValue(double x, double y, double z)
{
    vec[0] = x;
    vec[1] = y;
    vec[2] = z;
}

//=======================================================================

void Vec3d::setValue(Vec3d& v)
{
	vec[0] = v[0];
	vec[1] = v[1];
	vec[2] = v[2];
}

//=======================================================================

void Vec3d::setValue(double* val)
{
	vec[0] = val[0];
	vec[1] = val[1];
	vec[2] = val[2];
}
//=======================================================================

void Vec3d::normalize(double scale)
{
    double norm = sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
    if (norm != 0.0)
        norm = 1.0/norm;
    else
        norm = 1.0;
    vec[0] *= norm*scale;
    vec[1] *= norm*scale;
    vec[2] *= norm*scale;
}

//=======================================================================

double Vec3d::magnitude()
{
    return sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
}

double Vec3d::magnitude() const
{
    return sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
}

//=======================================================================

void Vec3d::print(const char *msg) const
{
	if (msg) {
    	printf("%s: %g, %g, %g\n", msg, vec[0], vec[1], vec[2]);
	} else {
    	printf("%g, %g, %g\n", vec[0], vec[1], vec[2]);
	}
    //theMsg->printf("%s: %g, %g, %g\n", msg, vec[0], vec[1], vec[2]);
}
#if 0
//----------------------------------------------------------------------
std::ostream&
operator<< (std::ostream&  os,
            const Vec3d& p)
{
//    os << '(' << p.x()  << ',' << p.y() << ',' << p.z() << ')';
    os << p.x() << ' ' << p.y() << ' ' << p.z(); 
    if (os.fail())
        cout << "operator<<(ostream&,Vec3d&) failed" << endl;
    return os;
}

//----------------------------------------------------------------------
std::istream&
operator>> (std::istream&  os,
            Vec3d& p)
{
    os >> p.x() >> p.y() >> p.z(); 
    if (os.fail())
        cout << "operator>>(istream&,Vec3d&) failed" << endl;
    return os;
}
#endif 
//----------------------------------------------------------------------
#ifdef STANDALONE

//void testvec(Vec3d& a)
//{
	//a.print("inside testvec Vec3d&, a= ");
//}
void testvec(Vec3d a)
{
	a.print("inside testvec Vec3d, a= ");
}

// Problems occur when I use testvec(Vec3d&) and 
// I allocate the vector on the stack. That is probably 
// because I cannot take a reference of such a vector.
// Therefore, one should either work with testvec(Vec3d) 
// or testvec(Vec3d&) but not both (for safety). It is also 
// safer not to allocate memory for the the arguments in
// place when calling the function IF the function argument 
// is a reference.

int main()
{
	Vec3d a(.2,.5,.7);
	Vec3d b(-.2,-.2,.8);

	Vec3d c;

	c = a + b;
	a.print("a= ");
	b.print("b= ");
	c.print("a+b");
	c = a - b;
	c.print("a-b");
	c = b - a;
	c.print("b-a");
	(a-b).print("inline a-b: ");

	Vec3d d = c  + b - 3*c;
	(a^b).print("a^b" );

    Vec3d dd = Vec3d(.2,.6,.9) + a;
	testvec(Vec3d(.2,.2,.2)+Vec3d(.1,.1,.1));
	testvec(dd^Vec3d(.2,.3,.5));
	testvec(dd += Vec3d(.3, .7. .2));

	return 0;
}
#endif
