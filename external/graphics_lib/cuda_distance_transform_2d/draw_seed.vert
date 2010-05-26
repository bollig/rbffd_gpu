//#extension GL_ARB_texture_rectangle : enable

// The vertex shader is screwing things up

varying vec4 curCol;
varying vec4 position;

uniform float szx; // 2D texture size
uniform float szy;
//----------------------------------------------------------------------
void main(void)
{
	// in that case, there is no need for model, etc. matrices
	// assume all coordinates between [0,1]
	// gl_Vertex: raw 3D coordinate
	// Transform to 2D coordinate in [0,1] x [0,1]

	curCol = gl_Color; // x,y,z in [0,1]

	// somehow, gl_Color does not have correct w value (i.e., the seed number)
	vec4 pos = gl_Vertex;  // 3D point in [0.,1.]^3
	vec2 p2 = vec2(szx,szy)*(pos.xy);
	//vec2 p2 = vec2(ip2);

	position = vec4(p2,0.,1.);
	gl_Position = gl_ModelViewProjectionMatrix * position;
}

// The z color is at the wrong location. 
// z is zero. So why is it at z=1 in framebuffer?
