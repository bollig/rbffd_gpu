#extension GL_ARB_texture_rectangle : enable

uniform sampler2D texture;
uniform float stepLength;
varying vec4 curPos;
varying vec4 texCoord01;
varying vec4 texCoord23;
varying vec4 texCoord56;
varying vec4 texCoord78;
float minDist; 

vec4 curSeed = vec4(1., 1., 1., 1.);

void func(vec4 val)
{
	vec2 offset = curPos.xy - val.xy; // GE
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
	//vec4 tx = texture2D(texture, gl_TexCoord[0].xy); // contains seed or large number

// simply calling the next line prevents the last line from being executed
// correctly. How is this possible?  texCoord is a varying variable. 
// Make sure that varying variables are NOT redefined inside the program

	vec4 tx = texture2D(texture, curPos.xy); // contains seed or large number

	// assumes texture is initialized so that distances are very large 
	curSeed.xy = tx.xy;
	minDist = 100.0;

	if (1 > 2) {
		gl_FragColor = vec4(1.,.5,0.,1.);
	} else {

	func(tx);

	val = texture2D(texture, texCoord01.xy); // closest seedpoint from neigh 0
	func(val); 
	val = texture2D(texture, texCoord01.zw); 
	func(val); 

	val = texture2D(texture, texCoord23.xy); // closest seedpoint from neigh 0
	func(val); 
	val = texture2D(texture, texCoord23.zw); 
	func(val); 

	val = texture2D(texture, texCoord56.xy); // closest seedpoint from neigh 0
	func(val); 
	val = texture2D(texture, texCoord56.zw); 
	func(val); 

	val = texture2D(texture, texCoord78.xy); // closest seedpoint from neigh 0
	func(val); 
	val = texture2D(texture, texCoord78.zw); 
	func(val); 


	// WHAT IS WRONG???? Clearly, curSeed.xy is not zero, else I would not 
	// see the picture of the voronoi mesh. 
	//gl_FragColor = vec4(curSeed.xx, 0, 1.);  // Clearly shows that it is NOT ZERO

	//vec2 off = curPos.xy - curSeed.xy;
	//vec2 off = curSeed.xy; // curSeed is black (NOT POSSIBLE)
	//float d = dot(off, off);
	//float d = curSeed.x*curSeed.x + curSeed.y*curSeed.y;
	gl_FragColor = vec4(curSeed.xy, 0.*minDist, 1.);
	//gl_FragColor = vec4(d,d, d, 1.);
	//gl_FragColor = vec4(minDist, minDist, minDist, 1.);

	} // end if
}
