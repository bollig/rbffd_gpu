#extension GL_ARB_texture_rectangle : enable

varying vec4 curCol;

void main(void)
{
	//gl_FragColor = curCol;
	gl_FragColor = vec4(0.,0.,0.,1.);
}
