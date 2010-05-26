#extension GL_ARB_texture_rectangle : enable

// For each fragment F, do the following: 
// 1. compute the distance between its position and that of the seed it contains
//    (initially contains the coordinates of a seed far away
// 2. for each neighbor, compute distance between its position and the seed 
//    contained in the neighbor
// 3. insert the seed with minimum distance into F

// Why isn't gl_TexCoord[0].xy the same as texCoord.xy?  //IMPORTANT
// vec4 tx = texture2DRect(texture, gl_TexCoord[0].xy); // contains seed or large number

// simply calling the next line prevents the last line from being executed
// correctly. How is this possible?  texCoord is a varying variable. 
// Make sure that varying variables are NOT redefined inside the program

uniform sampler2DRect texture;
uniform int stepLength;
uniform float tex_size;
uniform float szx;
uniform float szy;
uniform int sz3d; // 3d texture size

//uniform sampler1D k0_tex;

varying vec4 curPos;
varying vec2 q2m;
varying vec2 q20;
varying vec2 q2p;

varying float kf0;
int k0;

varying vec2 iv1;
varying vec2 iv2;


float minDist; 

vec4 point3d;
vec4 curPos3d; // in [0,1]

vec4 curSeed = vec4(0., 1., 1., 1.);
vec4 curSeed3d = vec4(0., 1., 1., 1.);

//----------------------------------------------------------------------
void tex_interp(vec2 p2) 
{
	// texture contains (x,y,z) in [0,1] coordinates of seed point
	vec4 seed = texture2DRect(texture, p2); // closest seedpoint from neigh 0

	vec3 offset = (curPos3d.xyz - seed.xyz); // GE
	float dist = dot(offset, offset);
	if (dist < minDist) {
		minDist = dist;
		curSeed3d = seed;
	}
}
//----------------------------------------------------------------------
void voronoi()
{
	vec2 p00 = vec2(-stepLength, -stepLength);
	vec2 p10 = vec2(         0., -stepLength);
	vec2 p20 = vec2( stepLength, -stepLength);
	vec2 p01 = vec2(-stepLength, 0.); 
	vec2 p11 = vec2(         0., 0.);
	vec2 p21 = vec2( stepLength, 0.);
	vec2 p02 = vec2(-stepLength,  stepLength);
	vec2 p12 = vec2(         0.,  stepLength);
	vec2 p22 = vec2( stepLength,  stepLength);

	vec2 p2;

	p2 = q2m + p00;
	tex_interp(p2);

	p2 = q2m + p10;
	tex_interp(p2);

	p2 = q2m + p20;
	tex_interp(p2);

	p2 = q2m + p01;
	tex_interp(p2);

	p2 = q2m + p11;
	tex_interp(p2);

	p2 = q2m + p21;
	tex_interp(p2);

	p2 = q2m + p02;
	tex_interp(p2);

	p2 = q2m + p12;
	tex_interp(p2);

	p2 = q2m + p22;
	tex_interp(p2);



	p2 = q20 + p00;
	tex_interp(p2);

	p2 = q20 + p10;
	tex_interp(p2);

	p2 = q20 + p20;
	tex_interp(p2);

	p2 = q20 + p01;
	tex_interp(p2);

	//p2 = q20 + p11;
	//tex_interp(p2);

	p2 = q20 + p21;
	tex_interp(p2);

	p2 = q20 + p02;
	tex_interp(p2);

	p2 = q20 + p12;
	tex_interp(p2);

	p2 = q20 + p22;
	tex_interp(p2);



	p2 = q2p + p00;
	tex_interp(p2);

	p2 = q2p + p10;
	tex_interp(p2);

	p2 = q2p + p20;
	tex_interp(p2);

	p2 = q2p + p01;
	tex_interp(p2);

	p2 = q2p + p11;
	tex_interp(p2);

	p2 = q2p + p21;
	tex_interp(p2);

	p2 = q2p + p02;
	tex_interp(p2);

	p2 = q2p + p12;
	tex_interp(p2);

	p2 = q2p + p22;
	tex_interp(p2);


	// WHY DOES IMAGE draw slowly to the screen

	gl_FragColor = vec4(curSeed3d.xyz, 1.);
}
//----------------------------------------------------------------------
void testcode()
{
	// curPos.xy : current position in rect tex units [0,szx] x [0,szy]
	// tx : current seed or (100,100,100) in range [0,1]^3
	//vec4 tx = texture2DRect(texture, curPos.xy); 
	curSeed3d = texture2DRect(texture, curPos.xy); 

	ivec2 ip2 = ivec2(curPos.xy);
	vec2 sz2 = vec2(szx, szy);
	ivec2 oxy;
	int gx = int(szx) / sz3d;
	k0 = int(kf0);
	oxy.y = k0 / gx;
	oxy.x = k0 - gx * oxy.y;

	// Current position in 3D space
	curPos3d.z = float(k0) / float(sz3d); // in [0,1]
	curPos3d.xy = (curPos.xy - vec2(sz3d*oxy)) / float(sz3d); // in [0,1]^2

	//vec3 off = (tx.xyz - curPos3d.xyz);
	vec3 off = (curSeed3d.xyz - curPos3d.xyz);
	minDist = dot(off, off);
	//curSeed3d.xyz = tx.xyz;
	curSeed3d.xyz = curSeed3d.xyz;

	//gl_FragColor = vec4(k0/float(sz3d));
	//gl_FragColor = vec4(curPos.xy/sz2, 0., 1.); // ok
	//gl_FragColor = vec4(vec2(oxy) / float(gx), 0., 1.); // not ok
	//gl_FragColor = vec4(oxy.x / float(gx));
	//gl_FragColor = vec4(oxy.y / float(gx), 0., 0., 1.);  // constant x, variable y
	//gl_FragColor = vec4(0., oxy.x / float(gx), 0., 1.);  // constant x, variable y
	//gl_FragColor = vec4(curPos3d.y); // WRONG

	//gl_FragColor.w = 1.0;

	//gl_FragColor.xyz = curSeed3d.xyz;

	voronoi();
}
//----------------------------------------------------------------------
void main(void)
{
	testcode();
	gl_FragColor = vec4(iv1, iv2);
}
//----------------------------------------------------------------------
