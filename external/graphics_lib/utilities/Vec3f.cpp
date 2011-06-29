
/*
 * Written by Dr. Gordon Erlebacher.
 */

#include <math.h>
#include <stdio.h>
#include <iostream>
//#include <Amira/HxMessage.h>
#include "Vec3f.h"

using namespace std;

//=======================================================================

#if 0
Vec3f::Vec3f()
{
    Vec3f((float)0., (float)0., (float)0.);
}

//=======================================================================

Vec3f::Vec3f(int x, int y, int z)
{
    vec[0] = (float) x;
    vec[1] = (float) y;
    vec[2] = (float) z;
}

//=======================================================================

Vec3f::Vec3f(float x, float y, float z)
{
    vec[0] = x;
    vec[1] = y;
    vec[2] = z;
}

//=======================================================================

Vec3f::Vec3f(double x, double y, double z)
{
    vec[0] = x;
    vec[1] = y;
    vec[2] = z;
}

//=======================================================================

Vec3f::Vec3f(float* pt)
{
    vec[0] = *pt++;
    vec[1] = *pt++;
    vec[2] = *pt;
}

//=======================================================================

Vec3f::Vec3f(double* pt)
{
    vec[0] = *pt++;
    vec[1] = *pt++;
    vec[2] = *pt;
}
#endif

//=======================================================================

Vec3f::Vec3f(Vec3f& vec)
{
    this->vec[0] = vec.x();
    this->vec[1] = vec.y();
    this->vec[2] = vec.z();
}

//=======================================================================

float* Vec3f::getVec()
{
    return vec;
}

//=======================================================================

void Vec3f::getVec(float* x, float* y, float* z)
{
    *x = vec[0];
    *y = vec[1];
    *z = vec[2];
}

//=======================================================================
void Vec3f::setValue(float x)
{
    vec[0] = x;
    vec[1] = x;
    vec[2] = x;
}
//=======================================================================

void Vec3f::setValue(float x, float y, float z)
{
    vec[0] = x;
    vec[1] = y;
    vec[2] = z;
}

//=======================================================================

void Vec3f::setValue(Vec3f& v)
{
	vec[0] = v[0];
	vec[1] = v[1];
	vec[2] = v[2];
}

//=======================================================================

void Vec3f::setValue(float* val)
{
	vec[0] = val[0];
	vec[1] = val[1];
	vec[2] = val[2];
}
//=======================================================================

void Vec3f::normalize(float scale)
{
    float norm = sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
    if (norm != 0.0)
        norm = 1.0/norm;
    else
        norm = 1.0;
    vec[0] *= norm*scale;
    vec[1] *= norm*scale;
    vec[2] *= norm*scale;
}

//=======================================================================

float Vec3f::magnitude()
{
    return sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
}

float Vec3f::magnitude() const
{
    return sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
}

//=======================================================================

void Vec3f::print(const char *msg) const
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
            const Vec3f& p)
{
//    os << '(' << p.x()  << ',' << p.y() << ',' << p.z() << ')';
    os << p.x() << ' ' << p.y() << ' ' << p.z(); 
    if (os.fail())
        cout << "operator<<(ostream&,Vec3f&) failed" << endl;
    return os;
}

//----------------------------------------------------------------------
std::istream&
operator>> (std::istream&  os,
            Vec3f& p)
{
    os >> p.x() >> p.y() >> p.z(); 
    if (os.fail())
        cout << "operator>>(istream&,Vec3f&) failed" << endl;
    return os;
}
#endif 
//----------------------------------------------------------------------
#ifdef STANDALONE

//void testvec(Vec3f& a)
//{
	//a.print("inside testvec Vec3f&, a= ");
//}
void testvec(Vec3f a)
{
	a.print("inside testvec Vec3f, a= ");
}

// Problems occur when I use testvec(Vec3f&) and 
// I allocate the vector on the stack. That is probably 
// because I cannot take a reference of such a vector.
// Therefore, one should either work with testvec(Vec3f) 
// or testvec(Vec3f&) but not both (for safety). It is also 
// safer not to allocate memory for the the arguments in
// place when calling the function IF the function argument 
// is a reference.

int main()
{
	Vec3f a(.2,.5,.7);
	Vec3f b(-.2,-.2,.8);

	Vec3f c;

	c = a + b;
	a.print("a= ");
	b.print("b= ");
	c.print("a+b");
	c = a - b;
	c.print("a-b");
	c = b - a;
	c.print("b-a");
	(a-b).print("inline a-b: ");

	Vec3f d = c  + b - 3*c;
	(a^b).print("a^b" );

    Vec3f dd = Vec3f(.2,.6,.9) + a;
	testvec(Vec3f(.2,.2,.2)+Vec3f(.1,.1,.1));
	testvec(dd^Vec3f(.2,.3,.5));
	testvec(dd += Vec3f(.3, .7. .2));

	return 0;
}
#endif
