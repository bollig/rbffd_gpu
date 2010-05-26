
project(graphics_utils)
cmake_minimum_required(VERSION 2.6)

set(src 
 abstract_textures.cpp  textfile.cpp Histogram.cpp gl_state.cpp platform.cpp 
 tex_ogl.cpp tex_ogl_1d.cpp textures.cpp utils.cpp textures_1d.cpp
 framebufferObject.cpp ping_pong.cpp abstract_textures_1d.cpp timege.cpp
)

find_path(glew_inc_path GL/glew.h /opt/local/include /panfs/panasas1/users/erlebach/vislab/src/glew/include/ /Developer/GPU_COMPUTING/C/common/inc/ DOC "path to GL/glew.h")

#ADD_DEFINITIONS(-DLINUX)

# enable for shader 4 configuration (e.g., geometry shader)
#ADD_DEFINITIONS(-DSHADER4)
#ADD_DEFINITIONS(-g)

# for mac
ADD_DEFINITIONS(-arch i386)

# for linux
#ADD_DEFINITIONS(-DLINUX)
#ADD_DEFINITIONS(-m64)


include_directories(
  ${glew_inc_path} 
  /Developer/GPU_Computing/shared/inc/
  ../utilities
  ../random
)

add_library(graphic_utilities STATIC ${src})

