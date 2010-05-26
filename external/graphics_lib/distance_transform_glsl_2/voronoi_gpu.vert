// The vertex shader is screwing things up

uniform float stepLength;
varying vec4 curPos;
varying vec4 texCoord01;
varying vec4 texCoord23;
varying vec4 texCoord56;
varying vec4 texCoord78;

void main(void)
{
	gl_Position = ftransform(); // modified by MODEL and PROJECTION matrices
	curPos = gl_MultiTexCoord0.xyxy;
	texCoord01 = curPos + vec4(-stepLength, -stepLength, 0, -stepLength);
	texCoord23 = curPos + vec4( stepLength, -stepLength, -stepLength, 0.);
	texCoord56 = curPos + vec4(-stepLength, 0, -stepLength, stepLength);
	texCoord78 = curPos + vec4(0, stepLength, stepLength, stepLength);
}
