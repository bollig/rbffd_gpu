#extension GL_ARB_texture_rectangle : enable

// The vertex shader is screwing things up

uniform float szx;
uniform float szy;
uniform int sz3d; // 3d texture size (cube)
uniform int stepLength;
//uniform float stepLength;
//uniform int km;
//uniform int kp;
//uniform int k0;


varying vec2 iv1;
varying vec2 iv2;

varying float kf0;
//float kf0;
int k0;

varying vec4 curPos;
varying vec2 q2m;
varying vec2 q20;
varying vec2 q2p;


//----------------------------------------------------------------------
// I will draw a series of quads (probably using ArrayBuffer objects)
// I need one ArrayBufferObject per texture level

// Input: a 2D point in the flat texture [0,texsize]
// pt -> 3D 
// compute 27 points

void main(void)
{
	gl_Position = ftransform(); // modified by MODEL and PROJECTION matrices
	curPos = gl_MultiTexCoord0.xyxy;  // in [0,szx] x [0,szy]
	kf0 = gl_MultiTexCoord0.z; // something wrong with z value
	k0 = int(kf0);

 		//ivec2 ip2 = ivec2(curPos.xy);
    	int gx = int(szx) / sz3d;
 		//ivec2 oxy = ip2/sz3d; // coord of tile origin that contains curPos

 		ivec2 oxy; 
 		ivec2 oxym;
 		ivec2 oxyp;

		int km = k0 - stepLength;
		int kp = k0 + stepLength;

		// above method does not work
		oxy.y = k0 / gx;
		oxy.x = k0 - oxy.y * gx;
		vec2 lxy = curPos.xy - vec2(oxy*sz3d);

		oxym.y = km / gx; // what if km < 0? (DEAL WITH THIS)
		oxyp.y = kp / gx;
		oxym.x = km - oxym.y * gx;
		oxyp.x = kp - oxyp.y * gx;

	q20 = lxy + vec2(oxy*sz3d); // location within the flat texture
 	q2m = lxy + vec2(oxym*sz3d);
 	q2p = lxy + vec2(oxyp*sz3d);

	iv1 = vec2(oxym);
	iv2 = vec2(oxyp);
}
//----------------------------------------------------------------------
