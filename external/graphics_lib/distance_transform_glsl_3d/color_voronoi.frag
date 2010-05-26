#extension GL_ARB_texture_rectangle : enable

uniform sampler2DRect tex;

void main(void)
{
	vec4 tx = gl_TexCoord[0] / 2000.;
	vec4 col = texture2DRect(tex, gl_TexCoord[0].xy);  // contains seed
	vec4 col1 = col*10.;
	gl_FragColor = vec4(fract(col1.x), fract(col1.y), 1.,1.);
}
