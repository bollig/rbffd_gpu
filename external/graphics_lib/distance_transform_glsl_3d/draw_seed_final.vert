#extension GL_ARB_texture_rectangle : enable

// The vertex shader is screwing things up

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

	ivec2 p2 = ivec2(gl_Vertex.xy * vec2(szx, szy));

	//ivec2 p2 = three2two(p3);
	vec2 p2f = vec2(p2) / vec2(szx, szy);

	vec4 pos = vec4(p2f,0.,1.);

	// default Ortho is (-1,1,-1,1);
	// Ortho2D is (0,1,0,1) (based on source file)
	gl_Position = gl_ModelViewProjectionMatrix * pos;

}
