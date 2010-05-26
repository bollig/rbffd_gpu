
#include <stdlib.h> // exit(0)
#include "glincludes.h"
#include "ping_pong_cuda.h"
#include "ping_pong.h"
#include "map.h"

using namespace std;

/// Create a Pingpong buffer
PingPongCuda::PingPongCuda(PingPong& ping) 
{
	pingpong = &ping;
	szx = ping.getWidth();
	szy = ping.getHeight();
	pbo = -1;
	vbo = -1;

	clock_tmp  = new GE::Time("PingPongCuda::misc");
	clock_tmp1 = new GE::Time("PingPongCuda::misc1");
	clock_tmp2 = new GE::Time("PingPongCuda::-misc2");
	clock_tmp3 = new GE::Time("PingPongCuda::misc3");

	// I might also go through VBO. In this case, I would use a second argument in 
	// the constructor using enums. 
	glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
	createPBO();
    //registerBufferObject(pbo);
}
//----------------------------------------------------------------------
PingPongCuda::~PingPongCuda()
{
	// pingpong object is the responsibility of the caller
    //unregisterBufferObject(pbo); // not clear when actually required
	glDeleteBuffers(1, &pbo);
	glDeleteBuffers(1, &vbo);
	pbo = vbo = -1;
}
//----------------------------------------------------------------------

void PingPongCuda::checkError(char* msg)   //   also sin superquadric
{
	printf("glerror: %s,  %s\n", msg, gluErrorString(glGetError())); // error
}
//----------------------------------------------------------------------
void PingPongCuda::toBackBuffer()
// unfortunately, the image on the screen appears to be 8 bit. 
// simply draw the texture to the screen. The second argument is not necessary.
{
	pingpong->toBackBuffer();
}
//----------------------------------------------------------------------
void PingPongCuda::print(int i1, int j1, int w, int h) 
{
	// Must be done outside PingPongCuda::begin()/end()
	PBO_to_FBO();
	pingpong->print(i1,j1,w,h);
}
//----------------------------------------------------------------------
void PingPongCuda::createPBO()
{
	//fbo_id = pingpong->getTexFBOid();

	//printf("CREATE PBO: szx, szy= %d, %d\n", szx, szy);
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
	// change 4 to take into account number of channels!!!
	glBufferData(GL_PIXEL_PACK_BUFFER, 4*szx*szy*sizeof(float), NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}
//----------------------------------------------------------------------
void PingPongCuda::FBO_to_PBO_faster()
{
// WHERE IS THE TIME LOST?
//
	printf("enter FBO_to_PBO_faster\n");
	clock_tmp->begin();


	TexOGL& tex = pingpong->getTexture();
	tex.bind(); // bind texture

	//clock_tmp1->begin();

	// CANNOT UNBIND pingpong at this stage, else data is not 
	// transferred correctly to the PBO
	// pingpong->unbind(); 

	// pack data into pbo
	// write data to the PBO
	glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo); // write into PBO
	//clock_tmp1->end();

	// perform the actual read (FBO MUST STILL BE ACTIVE)
	clock_tmp2->begin();
	//glReadPixels(0, 0, tex.getWidth(), tex.getHeight(), GL_RGBA, GL_FLOAT, 0);  // 13ms?
	glGetTexImage(tex.getTarget(), 0, GL_RGBA, GL_FLOAT, 0); // MUST I BIND TEXTURE?)
//	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, 0); // MUST I BIND TEXTURE?)
	clock_tmp2->end();

	//clock_tmp3->begin();
	// DISABLE PBO buffer
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0 );

	// DISABLE pingpong/FBO buffer
	//pingpong->unbind();
	//clock_tmp3->end();

	clock_tmp->end();

	clock_tmp->print();
	//clock_tmp1->print();
	clock_tmp2->print();
	//clock_tmp3->print();
	clock_tmp->reset();
	//clock_tmp1->reset();
	clock_tmp2->reset();
	//clock_tmp3->reset();
	printf("exit FBO_to_PBO_faster\n");
}
//----------------------------------------------------------------------
//----------------------------------------------------------------------
void PingPongCuda::FBO_to_PBO()
{
// WHERE IS THE TIME LOST?
//
	printf("enter FBO_to_PBO\n");
	clock_tmp->begin();
	pingpong->bind();   // bind FBO
	glReadBuffer(pingpong->getTexFBOid());

	//TexOGL& tex = pingpong->getTexture();
	//tex.bind(); // bind texture

	//clock_tmp1->begin();

	// CANNOT UNBIND pingpong at this stage, else data is not 
	// transferred correctly to the PBO
	// pingpong->unbind(); 

	// pack data into pbo
	// write data to the PBO
	glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo); // write into PBO
	//clock_tmp1->end();

	// perform the actual read (FBO MUST STILL BE ACTIVE)
	TexOGL& tex = pingpong->getTexture();
	//clock_tmp2->begin();
	glReadPixels(0, 0, tex.getWidth(), tex.getHeight(), GL_RGBA, GL_FLOAT, 0);  // 13ms?
	//glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, 0); // MUST I BIND TEXTURE?)
	//clock_tmp2->end();

	//clock_tmp3->begin();
	// DISABLE PBO buffer
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0 );

	// DISABLE pingpong/FBO buffer
	pingpong->unbind();
	//clock_tmp3->end();

	clock_tmp->end();

	clock_tmp->print();
//	clock_tmp1->print();
	//clock_tmp2->print();
	//clock_tmp3->print();
	clock_tmp->reset();
	//clock_tmp1->reset();
	//clock_tmp2->reset();
	//clock_tmp3->reset();

	printf("exit FBO_to_PBO\n");
}
//----------------------------------------------------------------------
void PingPongCuda::PBO_to_FBO()
{
	// precision is lost when transffering to FBO. WHY?? ERROR?
	//printf("PBO_to_FBO()\n");
	pingpong->setSubTexture(pbo, 0, 0, szx, szy);
}
//----------------------------------------------------------------------
// register PBO to CUDA
float* PingPongCuda::begin()
{
	FBO_to_PBO(); // uses glReadPixels()
	//FBO_to_PBO_faster(); // uses glGetTexImage();
	return cuda_register_and_map();
}
//----------------------------------------------------------------------
float* PingPongCuda::cuda_register_and_map()
{
    registerBufferObject(pbo);

	clock_tmp1->begin();
	mapBufferObject<float4>(&data4, pbo); // does this guarantee data4 is aligned?
	clock_tmp1->end();
	data = (float*) data4;

	clock_tmp1->print();
	clock_tmp1->reset();

	return data;
}
//----------------------------------------------------------------------
void PingPongCuda::end()
{
	clock_tmp2->begin();
	unmapBufferObject(pbo);
    unregisterBufferObject(pbo); // not clear when actually required
	clock_tmp2->end();
	clock_tmp2->print();
	clock_tmp2->reset();
}
//----------------------------------------------------------------------
// ASSUMES 4 components!!! But I need it to work with 2,3,4 components. 
// Automatically somehow
// If the results is zero or unexpected, it is possible that the data from the FBO 
// was not yet transferred to the PBO. This is done with begin() and end(9) pair.

void PingPongCuda::printPBO(int x0, int y0, int w, int h)
{
	// copy from PBO to user memory
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo); 
   	float* data_pbo = (float*) glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_READ_ONLY);
	//printf("data_pbo= %ld\n", (long) data_pbo);

	printf("\n-------- contents of PBO --------\n");

	// make sure I am not printing out of bounds
	TexOGL& tex = pingpong->getTexture();
	if ((x0+w) > tex.getWidth())  w = tex.getWidth()  - x0;
	if ((y0+h) > tex.getHeight()) h = tex.getHeight() - y0;

    for (int j=0; j < h; j++) {
    for (int i=0; i < w; i++) {
		int ix = w*(j+y0) + (i+x0);
        printf("data_pbo[%d,%d]: %f, %f, %f, %f\n", i,j, data_pbo[4*ix],
               data_pbo[4*ix+1], data_pbo[4*ix+2], data_pbo[4*ix+3]);
    }}
   	glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); 
}
//----------------------------------------------------------------------
