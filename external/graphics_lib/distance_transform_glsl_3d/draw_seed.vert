//#extension GL_ARB_texture_rectangle : enable

// The vertex shader is screwing things up

varying vec4 curCol;
varying vec4 position;
//vec4 position;

uniform int sz3d;
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
	vec4 pos = gl_Vertex;  // 3D point in [0.,1.]^3

	vec3 pos3 = pos.xyz * float(sz3d);
	ivec3 ip3 = ivec3(pos3); // [0,sz3d]^3

	ivec2 oxy;
	int gx = int(szx) / sz3d;
	oxy.y = ip3.z / gx;
	oxy.x = ip3.z - gx * oxy.y;
	ivec2 ip2 = sz3d*oxy + ip3.xy;

	//vec2 p2f = vec2(p2) / vec2(szx, szy); // in [0,1] x [0,1]
	vec2 p2 = vec2(ip2);
	position = vec4(p2,0.,1.);


	// default Ortho is (-1,1,-1,1);
	// Ortho2D is (0,1,0,1) (based on source file)
	// Location is correct, color is wrong. It is like I cannot write to the R component

	gl_Position = gl_ModelViewProjectionMatrix * position;
	//position = gl_Position;
}

// The z color is at the wrong location. 
// z is zero. So why is it at z=1 in framebuffer?
