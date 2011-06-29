
/*
 * Written by Dr. G. Erlebacher.
 */

#ifndef _VEC3_H_
#define _VEC3_H_

#include <math.h>

//class ostream;
#include <iostream> 
#include <vector>
// sizeof(Vec3) = 24

class Vec3
{
    public:

        inline Vec3()
			{ vec[0] = vec[1] = vec[2] = 0.0; };
        inline Vec3(const Vec3& v) {
            this->vec[0] = v.vec[0];
            this->vec[1] = v.vec[1];
            this->vec[2] = v.vec[2];
	    std::cout << ".";
        }
        inline Vec3(int x, int y=0, int z=0)
    		{ Vec3((double)x, (double)y, (double)z); }
        inline Vec3(float x, float y=0.f, float z=0.f)
			{ vec[0] = x; vec[1] = y; vec[2] = z; }
        inline Vec3(double x, double y=0., double z=0.)
			{ vec[0] = x; vec[1] = y; vec[2] = z; }
        /*
         * allocated in calling routine.
         */
		/// Memory is controlled by Vec3
        inline Vec3(float* pt)
			{ vec[0] = (double)pt[0]; vec[1] = (double)pt[1]; vec[2] = (double)pt[2]; }
		/// Memory is controlled by Vec3 (since Vec3 is float, and argument is double)
        inline Vec3(double* pt)
			{ vec[0] = pt[0]; vec[1] = pt[1]; vec[2] = pt[2]; }
        inline Vec3(const double* pt)
			{ vec[0] = pt[0]; vec[1] = pt[1]; vec[2] = pt[2]; }

       inline Vec3(std::vector<double> pt)
        {   
            if(pt.size() > 3) {
                std::cout << "Warning! Vec3 can only accept std::vector<double> with size 3!\n"; 
            } 
            for (int i = 0; i < pt.size() && i < 3; i++) 
                vec[i] = pt[i]; 
        }


        ~Vec3() {};
        Vec3(Vec3&);

        // Our dimension
        int size() const {
            return 3; 
        }

        int getDimension() const {
            return size(); 
        }

        double* data() {
            return vec; 
        }

        /*
         * memory allocated in class
         */
        double* getVec();
        void getVec(double* x, double* y, double* z);
        void setValue(double x, double y, double z=0.);
        void setValue(double x);
        void setValue(Vec3& v);
        void setValue(double* val);
        void normalize(double scale=1.0);
        double magnitude();
        double magnitude() const;
		// Euclidian distance squared
		double distance2(const Vec3 v) {
			return (v.x()-x())*(v.x()-x()) + 
			       (v.y()-y())*(v.y()-y()) + 
			       (v.z()-z())*(v.z()-z());
		}
		double square() {
			return (vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
		}

		//const double square() const {
		double square() const {
			return (vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
		}
        void print(const char *msg=0) const;
        //void print() {print("");}

        inline double x() {return vec[0];}
        inline double y() {return vec[1];}
        inline double z() {return vec[2];}

        inline double x() const {return vec[0];}
        inline double y() const {return vec[1];}
        inline double z() const {return vec[2];}

		// SHOULD BE STATIC
        Vec3 cross(const Vec3& a, const Vec3& b) {
            return (Vec3(a.vec[1]*b.vec[2]-a.vec[2]*b.vec[1],
                         a.vec[2]*b.vec[0]-a.vec[0]*b.vec[2],
                         a.vec[0]*b.vec[1]-a.vec[1]*b.vec[0]));
        }

        Vec3 cross(const Vec3& b) {
            return (Vec3(this->vec[1]*b.vec[2]-this->vec[2]*b.vec[1],
                         this->vec[2]*b.vec[0]-this->vec[0]*b.vec[2],
                         this->vec[0]*b.vec[1]-this->vec[1]*b.vec[0]));
		}
        /*
         * Overload operators, as needed.
         */
       const Vec3 operator+(const Vec3& a) const {
            return (Vec3(a.vec[0]+vec[0], a.vec[1]+vec[1], a.vec[2]+vec[2]));
	   }
       Vec3 operator+(Vec3& a) {
            return (Vec3(a.vec[0]+vec[0], a.vec[1]+vec[1], a.vec[2]+vec[2]));
	   }
       const Vec3 operator-(const Vec3& a) const {
           return (Vec3(vec[0]-a.vec[0], vec[1]-a.vec[1], vec[2]-a.vec[2]));
       }
       Vec3 operator-(Vec3& a) {
           return (Vec3(vec[0]-a.vec[0], vec[1]-a.vec[1], vec[2]-a.vec[2]));
       }
    	bool operator<(const Vec3& b) const {
        	return((vec[0] < b.vec[0]) && (vec[1] < b.vec[1]) && (vec[2] < b.vec[2]));
    	}
    	bool operator<=(const Vec3& b) const {
        	return((vec[0] <= b.vec[0]) && (vec[1] <= b.vec[1]) && (vec[2] <= b.vec[2]));
    	}
    	bool operator>(const Vec3& b) const {
        	return((vec[0] > b.vec[0]) && (vec[1] > b.vec[1]) && (vec[2] > b.vec[2]));
    	}
    	bool operator>=(const Vec3& b) const {
        	return((vec[0] >= b.vec[0]) && (vec[1] >= b.vec[1]) && (vec[2] >= b.vec[2]));
    	}
        friend Vec3 operator*(const Vec3& a, double f) {
            return (Vec3(a.vec[0]*f, a.vec[1]*f, a.vec[2]*f));
        }
        friend Vec3 operator*(double f, const Vec3& a) {
            return (Vec3(a.vec[0]*f, a.vec[1]*f, a.vec[2]*f));
        }
	   // GENERATES WARNING since a temporary is returned, and I might lose it. 
       //const Vec3& operator/(const Vec3& a) const {
	   // SOME COMPILERS MIGHT OPTIMIZE THIS TO AVOID COPIES!
       const Vec3 operator/(const Vec3& a) const {
           return (Vec3(vec[0]/a.vec[0], vec[1]/a.vec[1], vec[2]/a.vec[2]));
       }
	   // NOT ALLOWED TO RETURN NON-CONSTANT BECAUSE A TEMPORARY SHOULD NEVER BE CHANGED!
       Vec3 operator/(const Vec3& a) {
           return (Vec3(vec[0]/a.vec[0], vec[1]/a.vec[1], vec[2]/a.vec[2]));
       }
        /*
         * Addition: Brian M. Bouta
         */
        double operator*(const Vec3& a) const {
            return (a.vec[0]*vec[0] + a.vec[1]*vec[1] + a.vec[2]*vec[2]);
        }
        /*
         * Addition: Brian M. Bouta
         * The bivector formed from the outer product of two vectors
         * is treated as a vector, i.e., vec[0] = e1 ^ e2,
         * vec[1] = e2 ^ e3, vec[2] = e3 ^ e1.
         */
        Vec3 operator^(const Vec3& b) {
            return (Vec3(vec[0]*b.vec[1]-vec[1]*b.vec[0],
                         vec[1]*b.vec[2]-vec[2]*b.vec[1],
                         vec[2]*b.vec[0]-vec[0]*b.vec[2]));
        }
        /*
         * Addition: Brian M. Bouta
         */
        Vec3& operator+=(const Vec3& a)
        {
            vec[0] += a.vec[0];
            vec[1] += a.vec[1];
            vec[2] += a.vec[2];
            return *this;
        }


        Vec3& operator*=(double f)
        {
            vec[0] *= f;
            vec[1] *= f;
            vec[2] *= f;
            return *this;
        }
        Vec3& operator=(const Vec3& a) 
        {
            vec[0] = a.vec[0];
            vec[1] = a.vec[1];
            vec[2] = a.vec[2];
            return *this;
        }
		const double& operator[](const int i) const {
			return(vec[i]);
		}
		double operator()(int i) { // 1/5/08
			return(vec[i]);
		}
		double& operator[](int i) {
			return(vec[i]);
		}

		//friend Vec3 operator-(const Vec3&, const Vec3&);
		//friend Vec3 operator+(const Vec3&, const Vec3&);

	double cosine(Vec3& a, Vec3& b) {
    	double am = a.magnitude();
    	double bm = b.magnitude();
    	if ((fabs(am) < 1.e-7) || (fabs(bm) < 1.e-7)) {
        	return 1.0;
    	}

    	return ((a*b)/(am*bm));
	}

	int isColinear(Vec3& a, Vec3& b)
	{
    	double am = a.magnitude();
    	double bm = b.magnitude();
    	if ((fabs(am) < 1.e-7) || (fabs(bm) < 1.e-7)) {
        	return 0;
    	}
	
    	if (fabs((a*b)/(am*bm)-1.0) < 1.e-5) {
        	return 1;
    	} else {
        	return 0; // not collinear
   		}
	}
	int isZero(double tolerance)
	{
		if ((vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]) 
				< (tolerance*tolerance)) {
			return 1;
		} else {
			return 0;
		}
	}

	// Allow output to C++ streams
	friend std::ostream& operator<< (std::ostream& os, const Vec3& p) {
		//    os << '(' << p.x()  << ',' << p.y() << ',' << p.z() << ')';

        os.setf(std::ios::fixed, std::ios::floatfield); 
        os.setf(std::ios::showpoint); 
        os.precision( 8 );   // 8 Digits because our Vec3 is actually a Vec3f 
        os.width(13);
		os << std::right << p.x() ;
        os.width(13);
        os << std::right << p.y();
        os.width(13);
        os << std::right << p.z();
		if (os.fail())
		      std::cout << "operator<<(ostream&,Vec3&) failed" << std::endl;
		return os;
	}
	friend std::istream& operator>> (std::istream& is, Vec3& p) {
		//is >> p.x() >> p.y() >> p.z();
		double x,y,z; 
		
		is >> x >> y >> z; 

		p[0] = x; 
		p[1] = y; 
		p[2] = z;

	//	if (is.eof())
	//		std::cout << "operator>>(istream&, Vec3&) reached EOF" << std::endl;
	//	if (is.fail())
	//		std::cout << "operator>>(istream&,Vec3&) failed" << std::endl;
		return is;
	}
//----------------------------------------------------------------------

    public:
        double vec[3];
};



#if 0
Vec3 operator-(const Vec3& a, const Vec3& b) {
	return (Vec3(a.vec[0]-b.vec[0], a.vec[1]-b.vec[1], a.vec[2]-b.vec[2]));
}

Vec3 operator+(Vec3& a, Vec3& b) {
	return (Vec3(a.x()+b.x(), a.y()+b.y(), a.z()+b.z()));
}
#endif

#endif

