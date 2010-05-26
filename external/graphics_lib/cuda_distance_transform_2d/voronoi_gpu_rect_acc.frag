#extension GL_ARB_texture_rectangle : enable

uniform sampler2DRect texture;
uniform float stepLength;
uniform float tex_size;
varying vec4 curPos;
varying vec4 texCoord01;
varying vec4 texCoord23;
varying vec4 texCoord56;
varying vec4 texCoord78;
float minDist; 

vec4 curSeed = vec4(0., 1., 1., 1.);

void func(vec4 val)
{
	vec2 offset = (curPos.xy/tex_size - val.xy); // GE
	float dist = dot(offset, offset);
	if (dist < minDist) {
		minDist = dist;
		curSeed = val;
	}
}

void main(void)
{
	vec4 val;

	// For each fragment F, do the following: 
	// 1. compute the distance between its position and that of the seed it contains
	//    (initially contains the coordinates of a seed far away
	// 2. for each neighbor, compute distance between its position and the seed 
	//    contained in the neighbor
	// 3. insert the seed with minimum distance into F

	// Why isn't gl_TexCoord[0].xy the same as texCoord.xy?  //IMPORTANT
	//vec4 tx = texture2DRect(texture, gl_TexCoord[0].xy); // contains seed or large number

// simply calling the next line prevents the last line from being executed
// correctly. How is this possible?  texCoord is a varying variable. 
// Make sure that varying variables are NOT redefined inside the program

	// number in [0,tex_size[
	vec4 tx = texture2DRect(texture, curPos.xy); // contains seed [0,1] or large number

	vec2 off = (tx.xy - curPos.xy/tex_size);
	minDist = dot(off, off);

	// assumes texture is initialized so that distances are very large 
	curSeed = tx;

	//if (1 > 2) {
		//gl_FragColor = vec4(1.,.5,0.,1.);
	//} else {


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


	//gl_FragColor = vec4(curSeed.xy, 0., 1.);
	//gl_FragColor = vec4(curSeed.xy, curSeed.x * curSeed.y, 1.);
	//gl_FragColor = vec4(1.,1.,1.,1.);
	//gl_FragColor = vec4(curSeed.xy, 1.-200.*minDist, 1.);
	//gl_FragColor = vec4(tx.xy, 1.-minDist, 1.);

	curSeed.z = 0.;
	gl_FragColor = curSeed; // last component is seed number
	//gl_FragColor = vec4(0.5,0.5,1.,.2);

	//} // end if
}
