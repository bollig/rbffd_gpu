
/*
 * Written by Dr. G. Erlebacher.
 */

#ifndef _VEC3F_H_
#define _VEC3F_H_

#include <math.h>

//class ostream;
#include <iostream> 
#include <vector>
// sizeof(Vec3f) = 24

class Vec3f
{
    public:

        inline Vec3f()
			{ vec[0] = vec[1] = vec[2] = 0.0; };
        inline Vec3f(const Vec3f& v) {
            this->vec[0] = v.vec[0];
            this->vec[1] = v.vec[1];
            this->vec[2] = v.vec[2];
	  //  std::cout << ".";
        }
        inline Vec3f(int x, int y=0, int z=0)
    		{ Vec3f((float)x, (float)y, (float)z); }
        inline Vec3f(float x, float y=0.f, float z=0.f)
			{ vec[0] = x; vec[1] = y; vec[2] = z; }
        inline Vec3f(double x, double y=0., double z=0.)
			{ vec[0] = x; vec[1] = y; vec[2] = z; }
        /*
         * allocated in calling routine.
         */
		/// Memory is controlled by Vec3f
        inline Vec3f(float* pt)
			{ vec[0] = pt[0]; vec[1] = pt[1]; vec[2] = pt[2]; }
		/// Memory is controlled by Vec3f (since Vec3f is float, and argument is double)
        inline Vec3f(double* pt)
			{ vec[0] = pt[0]; vec[1] = pt[1]; vec[2] = pt[2]; }
        inline Vec3f(const double* pt)
			{ vec[0] = pt[0]; vec[1] = pt[1]; vec[2] = pt[2]; }

       inline Vec3f(std::vector<double> pt)
        {   
            if(pt.size() > 3) {
                std::cout << "Warning! Vec3f can only accept std::vector<double> with size 3!\n"; 
            } 
            for (int i = 0; i < pt.size() && i < 3; i++) 
                vec[i] = pt[i]; 
        }


        ~Vec3f() {};
        Vec3f(Vec3f&);

        // Our dimension
        int size() const {
            return 3; 
        }

        int getDimension() const {
            return size(); 
        }

        float* data() {
            return vec; 
        }

        /*
         * memory allocated in class
         */
        float* getVec();
        void getVec(float* x, float* y, float* z);
        void setValue(float x, float y, float z=0.);
        void setValue(float x);
        void setValue(Vec3f& v);
        void setValue(float* val);
        void normalize(float scale=1.0);
        float magnitude();
        float magnitude() const;
		// Euclidian distance squared
		double distance2(const Vec3f v) {
			return (v.x()-x())*(v.x()-x()) + 
			       (v.y()-y())*(v.y()-y()) + 
			       (v.z()-z())*(v.z()-z());
		}
		float square() {
			return (vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
		}

		//const float square() const {
		float square() const {
			return (vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
		}
        void print(const char *msg=0) const;
        //void print() {print("");}

        inline float x() {return vec[0];}
        inline float y() {return vec[1];}
        inline float z() {return vec[2];}

        inline float x() const {return vec[0];}
        inline float y() const {return vec[1];}
        inline float z() const {return vec[2];}

		// SHOULD BE STATIC
        Vec3f cross(const Vec3f& a, const Vec3f& b) {
            return (Vec3f(a.vec[1]*b.vec[2]-a.vec[2]*b.vec[1],
                         a.vec[2]*b.vec[0]-a.vec[0]*b.vec[2],
                         a.vec[0]*b.vec[1]-a.vec[1]*b.vec[0]));
        }

        Vec3f cross(const Vec3f& b) {
            return (Vec3f(this->vec[1]*b.vec[2]-this->vec[2]*b.vec[1],
                         this->vec[2]*b.vec[0]-this->vec[0]*b.vec[2],
                         this->vec[0]*b.vec[1]-this->vec[1]*b.vec[0]));
		}
        /*
         * Overload operators, as needed.
         */
       const Vec3f operator+(const Vec3f& a) const {
            return (Vec3f(a.vec[0]+vec[0], a.vec[1]+vec[1], a.vec[2]+vec[2]));
	   }
       Vec3f operator+(Vec3f& a) {
            return (Vec3f(a.vec[0]+vec[0], a.vec[1]+vec[1], a.vec[2]+vec[2]));
	   }
       const Vec3f operator-(const Vec3f& a) const {
           return (Vec3f(vec[0]-a.vec[0], vec[1]-a.vec[1], vec[2]-a.vec[2]));
       }
       Vec3f operator-(Vec3f& a) {
           return (Vec3f(vec[0]-a.vec[0], vec[1]-a.vec[1], vec[2]-a.vec[2]));
       }
    	bool operator<(const Vec3f& b) const {
        	return((vec[0] < b.vec[0]) && (vec[1] < b.vec[1]) && (vec[2] < b.vec[2]));
    	}
    	bool operator<=(const Vec3f& b) const {
        	return((vec[0] <= b.vec[0]) && (vec[1] <= b.vec[1]) && (vec[2] <= b.vec[2]));
    	}
    	bool operator>(const Vec3f& b) const {
        	return((vec[0] > b.vec[0]) && (vec[1] > b.vec[1]) && (vec[2] > b.vec[2]));
    	}
    	bool operator>=(const Vec3f& b) const {
        	return((vec[0] >= b.vec[0]) && (vec[1] >= b.vec[1]) && (vec[2] >= b.vec[2]));
    	}
        friend Vec3f operator*(const Vec3f& a, double f) {
            return (Vec3f(a.vec[0]*f, a.vec[1]*f, a.vec[2]*f));
        }
        friend Vec3f operator*(double f, const Vec3f& a) {
            return (Vec3f(a.vec[0]*f, a.vec[1]*f, a.vec[2]*f));
        }
	   // GENERATES WARNING since a temporary is returned, and I might lose it. 
       //const Vec3f& operator/(const Vec3f& a) const {
	   // SOME COMPILERS MIGHT OPTIMIZE THIS TO AVOID COPIES!
       const Vec3f operator/(const Vec3f& a) const {
           return (Vec3f(vec[0]/a.vec[0], vec[1]/a.vec[1], vec[2]/a.vec[2]));
       }
	   // NOT ALLOWED TO RETURN NON-CONSTANT BECAUSE A TEMPORARY SHOULD NEVER BE CHANGED!
       Vec3f operator/(const Vec3f& a) {
           return (Vec3f(vec[0]/a.vec[0], vec[1]/a.vec[1], vec[2]/a.vec[2]));
       }
        /*
         * Addition: Brian M. Bouta
         */
        float operator*(const Vec3f& a) const {
            return (a.vec[0]*vec[0] + a.vec[1]*vec[1] + a.vec[2]*vec[2]);
        }
        /*
         * Addition: Brian M. Bouta
         * The bivector formed from the outer product of two vectors
         * is treated as a vector, i.e., vec[0] = e1 ^ e2,
         * vec[1] = e2 ^ e3, vec[2] = e3 ^ e1.
         */
        Vec3f operator^(const Vec3f& b) {
            return (Vec3f(vec[0]*b.vec[1]-vec[1]*b.vec[0],
                         vec[1]*b.vec[2]-vec[2]*b.vec[1],
                         vec[2]*b.vec[0]-vec[0]*b.vec[2]));
        }
        /*
         * Addition: Brian M. Bouta
         */
        Vec3f& operator+=(const Vec3f& a)
        {
            vec[0] += a.vec[0];
            vec[1] += a.vec[1];
            vec[2] += a.vec[2];
            return *this;
        }


        Vec3f& operator*=(float f)
        {
            vec[0] *= f;
            vec[1] *= f;
            vec[2] *= f;
            return *this;
        }
        Vec3f& operator=(const Vec3f& a) 
        {
            vec[0] = a.vec[0];
            vec[1] = a.vec[1];
            vec[2] = a.vec[2];
            return *this;
        }
		const float& operator[](const int i) const {
			return(vec[i]);
		}
		float operator()(int i) { // 1/5/08
			return(vec[i]);
		}
		float& operator[](int i) {
			return(vec[i]);
		}

		//friend Vec3f operator-(const Vec3f&, const Vec3f&);
		//friend Vec3f operator+(const Vec3f&, const Vec3f&);

	float cosine(Vec3f& a, Vec3f& b) {
    	float am = a.magnitude();
    	float bm = b.magnitude();
    	if ((fabs(am) < 1.e-7) || (fabs(bm) < 1.e-7)) {
        	return 1.0;
    	}

    	return ((a*b)/(am*bm));
	}

	int isColinear(Vec3f& a, Vec3f& b)
	{
    	float am = a.magnitude();
    	float bm = b.magnitude();
    	if ((fabs(am) < 1.e-7) || (fabs(bm) < 1.e-7)) {
        	return 0;
    	}
	
    	if (fabs((a*b)/(am*bm)-1.0) < 1.e-5) {
        	return 1;
    	} else {
        	return 0; // not collinear
   		}
	}
	int isZero(float tolerance)
	{
		if ((vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]) 
				< (tolerance*tolerance)) {
			return 1;
		} else {
			return 0;
		}
	}

	// Allow output to C++ streams
	friend std::ostream& operator<< (std::ostream& os, const Vec3f& p) {
		//    os << '(' << p.x()  << ',' << p.y() << ',' << p.z() << ')';

        os.setf(std::ios::fixed, std::ios::floatfield); 
        os.setf(std::ios::showpoint); 
        os.precision( 8 );   // 8 Digits because our Vec3f is actually a Vec3ff 
        os.width(13);
		os << std::right << p.x() ;
        os.width(13);
        os << std::right << p.y();
        os.width(13);
        os << std::right << p.z();
		if (os.fail())
		      std::cout << "operator<<(ostream&,Vec3f&) failed" << std::endl;
		return os;
	}
	friend std::istream& operator>> (std::istream& is, Vec3f& p) {
		//is >> p.x() >> p.y() >> p.z();
		float x,y,z; 
		
		is >> x >> y >> z; 

		p[0] = x; 
		p[1] = y; 
		p[2] = z;

	//	if (is.eof())
	//		std::cout << "operator>>(istream&, Vec3f&) reached EOF" << std::endl;
	//	if (is.fail())
	//		std::cout << "operator>>(istream&,Vec3f&) failed" << std::endl;
		return is;
	}
//----------------------------------------------------------------------

    public:
        float vec[3];
};



#if 0
Vec3f operator-(const Vec3f& a, const Vec3f& b) {
	return (Vec3f(a.vec[0]-b.vec[0], a.vec[1]-b.vec[1], a.vec[2]-b.vec[2]));
}

Vec3f operator+(Vec3f& a, Vec3f& b) {
	return (Vec3f(a.x()+b.x(), a.y()+b.y(), a.z()+b.z()));
}
#endif

#endif

