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
uniform float stepLength;
uniform float tex_size;
uniform float szx;
uniform float szy;
uniform int sz3d; // 3d texture size

varying vec4 curPos;
varying vec4 texCoord01;
varying vec4 texCoord23;
varying vec4 texCoord56;
varying vec4 texCoord78;
float minDist; 

vec4 point3d;
vec4 curPos3d; // in [0,1]

vec4 curSeed = vec4(0., 1., 1., 1.);
vec4 curSeed3d = vec4(0., 1., 1., 1.);


//----------------------------------------------------------------------
ivec2 three2two(ivec3 p3) // coord in range [0,n-1]
{
	ivec2 oxy;
	int gx = int(szx) / sz3d;
	oxy.y = p3.z / gx;
	oxy.x = p3.z - gx*oxy.y;
	return (sz3d*oxy + p3.xy);
}
//----------------------------------------------------------------------
// coord in range [0,n-1]
ivec3 two2three(ivec2 ip2) 
{
   	int gx = int(szx) / sz3d; // #tiles in x direction
	ivec2 oxy = ip2 / sz3d;    // which tile (along x and y)
   	ivec3 ip3;                 // 3D coordinate
   	ip3.xy = ip2 - oxy*sz3d;    // xy coordinate relative to tile origin
   	ip3.z  = oxy.x + oxy.y * gx;
   	return ip3;
}
//----------------------------------------------------------------------
// input 2D coordinates in [0, tex_size[
// output 3D coordinates in [0,1]
vec3 two2threef(vec2 xy)
{
	ivec2 ip2 = ivec2(xy);
	ivec3 ip3 = two2three(ip2);
	float ss = 1. / float(sz3d);
	vec3 p3 = vec3(ip3) * vec3(ss); //  / float(sz3d);
	return p3; // in range [0.,1.]
}
//----------------------------------------------------------------------
void func(vec4 val)
{
	vec3 offset = (curPos3d.xyz - val.xyz); // GE
	float dist = dot(offset, offset);
	if (dist < minDist) {
		minDist = dist;
		curSeed3d = val;
	}
}
//----------------------------------------------------------------------

void main(void)
{
	vec4 val;

	// vec2 curPos :  [0,tex_size[
	// vec4 tx,  tx.xyz :  [0,1]
	vec4 tx = texture2DRect(texture, curPos.xy); // contains seed [0,1] or large number

	// I need the current position in the 3D grid, from the 2D position 
	// in the 2D texture (only need this once). 
	// input  in [0,tex_size[
	// output in [0,1]

	// curPos.xy in [0, tex_size[ (texture coordinates of flat texture)
	// curPos3d.xyz in [0,1]

	curPos3d.xyz = two2threef(curPos.xy); 

	// assumes texture is initialized so that distances are very large 
	vec3 off = (tx.xyz - curPos3d.xyz);
	minDist = dot(off, off);
	curSeed3d.xyz = tx.xyz;

	// (x,y) are large numbers on the first pass
	val = texture2DRect(texture, texCoord01.xy); // closest seedpoint from neigh 0
	func(val); 
	val = texture2DRect(texture, texCoord01.zw); 
	func(val); 

	val = texture2DRect(texture, texCoord23.xy); // closest seedpoint from neigh 0
	func(val); 
	val = texture2DRect(texture, texCoord23.zw); 
	func(val); 

	val = texture2DRect(texture, texCoord56.xy); // closest seedpoint from neigh 0
	func(val); 
	val = texture2DRect(texture, texCoord56.zw); 
	func(val); 

	val = texture2DRect(texture, texCoord78.xy); // closest seedpoint from neigh 0
	func(val); 
	val = texture2DRect(texture, texCoord78.zw); 
	func(val); 


	// WHY DOES IMAGE draw slowly to the screen
	gl_FragColor = vec4(curSeed3d.xyz, 1.);

	//gl_FragColor = vec4(1.,0.,1.,1.);
	//gl_FragColor = vec4(curSeed.xy, 1.-200.*minDist, 1.);
	//gl_FragColor = vec4(tx.xy, 1.-minDist, 1.);

}
