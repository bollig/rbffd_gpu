//#extension GL_ARB_texture_rectangle : enable

// test dependent texture read to test floating point interpolation

uniform sampler2D screen_tex;

void main(void)
{
	vec2 pos0 = gl_TexCoord[0].xy;
  	gl_FragColor = texture2D(screen_tex, pos0); // works except wrong texture
	gl_FragColor = vec4(1.,0.,0.,0.);
}
