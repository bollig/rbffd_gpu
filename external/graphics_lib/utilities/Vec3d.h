
/*
 * Written by Dr. G. Erlebacher.
 */

#ifndef _VEC3D_H_
#define _VEC3D_H_

#include <math.h>

//class ostream;
#include <iostream> 
#include <vector>
// sizeof(Vec3d) = 24

class Vec3d
{
    public:

        inline Vec3d()
			{ vec[0] = vec[1] = vec[2] = 0.0; };
        inline Vec3d(const Vec3d& v) {
            this->vec[0] = v.vec[0];
            this->vec[1] = v.vec[1];
            this->vec[2] = v.vec[2];
//	    std::cout << ".";
        }
        inline Vec3d(int x, int y=0, int z=0)
    		{ Vec3d((double)x, (double)y, (double)z); }
        inline Vec3d(float x, float y=0.f, float z=0.f)
			{ vec[0] = x; vec[1] = y; vec[2] = z; }
        inline Vec3d(double x, double y=0., double z=0.)
			{ vec[0] = x; vec[1] = y; vec[2] = z; }
        /*
         * allocated in calling routine.
         */
		/// Memory is controlled by Vec3d
        inline Vec3d(float* pt)
			{ vec[0] = (double)pt[0]; vec[1] = (double)pt[1]; vec[2] = (double)pt[2]; }
		/// Memory is controlled by Vec3d (since Vec3d is float, and argument is double)
        inline Vec3d(double* pt)
			{ vec[0] = pt[0]; vec[1] = pt[1]; vec[2] = pt[2]; }
        inline Vec3d(const double* pt)
			{ vec[0] = pt[0]; vec[1] = pt[1]; vec[2] = pt[2]; }

       inline Vec3d(std::vector<double> pt)
        {   
            if(pt.size() > 3) {
                std::cout << "Warning! Vec3d can only accept std::vector<double> with size 3!\n"; 
            } 
            for (int i = 0; i < pt.size() && i < 3; i++) 
                vec[i] = pt[i]; 
        }


        ~Vec3d() {};
        Vec3d(Vec3d&);

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
        void setValue(Vec3d& v);
        void setValue(double* val);
        void normalize(double scale=1.0);
        double magnitude();
        double magnitude() const;
		// Euclidian distance squared
		double distance2(const Vec3d v) {
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
        Vec3d cross(const Vec3d& a, const Vec3d& b) {
            return (Vec3d(a.vec[1]*b.vec[2]-a.vec[2]*b.vec[1],
                         a.vec[2]*b.vec[0]-a.vec[0]*b.vec[2],
                         a.vec[0]*b.vec[1]-a.vec[1]*b.vec[0]));
        }

        Vec3d cross(const Vec3d& b) {
            return (Vec3d(this->vec[1]*b.vec[2]-this->vec[2]*b.vec[1],
                         this->vec[2]*b.vec[0]-this->vec[0]*b.vec[2],
                         this->vec[0]*b.vec[1]-this->vec[1]*b.vec[0]));
		}
        /*
         * Overload operators, as needed.
         */
       const Vec3d operator+(const Vec3d& a) const {
            return (Vec3d(a.vec[0]+vec[0], a.vec[1]+vec[1], a.vec[2]+vec[2]));
	   }
       Vec3d operator+(Vec3d& a) {
            return (Vec3d(a.vec[0]+vec[0], a.vec[1]+vec[1], a.vec[2]+vec[2]));
	   }
       const Vec3d operator-(const Vec3d& a) const {
           return (Vec3d(vec[0]-a.vec[0], vec[1]-a.vec[1], vec[2]-a.vec[2]));
       }
       Vec3d operator-(Vec3d& a) {
           return (Vec3d(vec[0]-a.vec[0], vec[1]-a.vec[1], vec[2]-a.vec[2]));
       }
    	bool operator<(const Vec3d& b) const {
        	return((vec[0] < b.vec[0]) && (vec[1] < b.vec[1]) && (vec[2] < b.vec[2]));
    	}
    	bool operator<=(const Vec3d& b) const {
        	return((vec[0] <= b.vec[0]) && (vec[1] <= b.vec[1]) && (vec[2] <= b.vec[2]));
    	}
    	bool operator>(const Vec3d& b) const {
        	return((vec[0] > b.vec[0]) && (vec[1] > b.vec[1]) && (vec[2] > b.vec[2]));
    	}
    	bool operator>=(const Vec3d& b) const {
        	return((vec[0] >= b.vec[0]) && (vec[1] >= b.vec[1]) && (vec[2] >= b.vec[2]));
    	}
        friend Vec3d operator*(const Vec3d& a, double f) {
            return (Vec3d(a.vec[0]*f, a.vec[1]*f, a.vec[2]*f));
        }
        friend Vec3d operator*(double f, const Vec3d& a) {
            return (Vec3d(a.vec[0]*f, a.vec[1]*f, a.vec[2]*f));
        }
	   // GENERATES WARNING since a temporary is returned, and I might lose it. 
       //const Vec3d& operator/(const Vec3d& a) const {
	   // SOME COMPILERS MIGHT OPTIMIZE THIS TO AVOID COPIES!
       const Vec3d operator/(const Vec3d& a) const {
           return (Vec3d(vec[0]/a.vec[0], vec[1]/a.vec[1], vec[2]/a.vec[2]));
       }
	   // NOT ALLOWED TO RETURN NON-CONSTANT BECAUSE A TEMPORARY SHOULD NEVER BE CHANGED!
       Vec3d operator/(const Vec3d& a) {
           return (Vec3d(vec[0]/a.vec[0], vec[1]/a.vec[1], vec[2]/a.vec[2]));
       }
        /*
         * Addition: Brian M. Bouta
         */
        double operator*(const Vec3d& a) const {
            return (a.vec[0]*vec[0] + a.vec[1]*vec[1] + a.vec[2]*vec[2]);
        }
        /*
         * Addition: Brian M. Bouta
         * The bivector formed from the outer product of two vectors
         * is treated as a vector, i.e., vec[0] = e1 ^ e2,
         * vec[1] = e2 ^ e3, vec[2] = e3 ^ e1.
         */
        Vec3d operator^(const Vec3d& b) {
            return (Vec3d(vec[0]*b.vec[1]-vec[1]*b.vec[0],
                         vec[1]*b.vec[2]-vec[2]*b.vec[1],
                         vec[2]*b.vec[0]-vec[0]*b.vec[2]));
        }
        /*
         * Addition: Brian M. Bouta
         */
        Vec3d& operator+=(const Vec3d& a)
        {
            vec[0] += a.vec[0];
            vec[1] += a.vec[1];
            vec[2] += a.vec[2];
            return *this;
        }


        Vec3d& operator*=(double f)
        {
            vec[0] *= f;
            vec[1] *= f;
            vec[2] *= f;
            return *this;
        }
        Vec3d& operator=(const Vec3d& a) 
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

		//friend Vec3d operator-(const Vec3d&, const Vec3d&);
		//friend Vec3d operator+(const Vec3d&, const Vec3d&);

	double cosine(Vec3d& a, Vec3d& b) {
    	double am = a.magnitude();
    	double bm = b.magnitude();
    	if ((fabs(am) < 1.e-7) || (fabs(bm) < 1.e-7)) {
        	return 1.0;
    	}

    	return ((a*b)/(am*bm));
	}

	int isColinear(Vec3d& a, Vec3d& b)
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
	friend std::ostream& operator<< (std::ostream& os, const Vec3d& p) {
		//    os << '(' << p.x()  << ',' << p.y() << ',' << p.z() << ')';

        os.setf(std::ios::fixed, std::ios::floatfield); 
        os.setf(std::ios::showpoint); 
        os.precision( 15 );   
        os.width(20);
		os << std::right << p.x() ;
        os.width(20);
        os << std::right << p.y();
        os.width(20);
        os << std::right << p.z();
		if (os.fail())
		      std::cout << "operator<<(ostream&,Vec3d&) failed" << std::endl;
		return os;
	}
	friend std::istream& operator>> (std::istream& is, Vec3d& p) {
		//is >> p.x() >> p.y() >> p.z();
		double x,y,z; 
		
		is >> x >> y >> z; 

		p[0] = x; 
		p[1] = y; 
		p[2] = z;

	//	if (is.eof())
	//		std::cout << "operator>>(istream&, Vec3d&) reached EOF" << std::endl;
	//	if (is.fail())
	//		std::cout << "operator>>(istream&,Vec3d&) failed" << std::endl;
		return is;
	}
//----------------------------------------------------------------------

    public:
        double vec[3];
};



#if 0
Vec3d operator-(const Vec3d& a, const Vec3d& b) {
	return (Vec3d(a.vec[0]-b.vec[0], a.vec[1]-b.vec[1], a.vec[2]-b.vec[2]));
}

Vec3d operator+(Vec3d& a, Vec3d& b) {
	return (Vec3d(a.x()+b.x(), a.y()+b.y(), a.z()+b.z()));
}
#endif

#endif

