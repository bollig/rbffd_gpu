#extension GL_ARB_texture_rectangle : enable

varying vec4 curCol;
varying vec4 position;

void main(void)
{
	//gl_FragColor = vec4(0.75, 0.65, 0.55, 1.);
	//gl_FragColor = vec4(0.5,0.5,0.5,0.5);
	//gl_FragColor = position;

 	gl_FragColor = curCol; // [12,6]
}
