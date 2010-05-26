
project(hdw_test)
# use rect textures but accelerated algorithm
set(src main.cpp globals.cpp distance_transform_rect_acc.cpp )
# use rect textures
#set(src main.cpp globals.cpp distance_transform_rect.cpp )
# use 2D textures
#set(src main.cpp globals.cpp distance_transform.cpp )

# SOME HARD-CODED PATHS
find_library(glew_lib GLEW PATHS /opt/local/lib DOC "glew library path")
find_library(utilities utilities PATHS ../utilities)
find_library(graphics graphic_utilities PATHS ../graphics_utils)
find_library(random random PATHS ../random)
find_library(timing timer PATHS ../timing)
find_path(glew_inc_path GL/glew.h /opt/local/include DOC "path to GL/glew.h")

INCLUDE(FindOpenGL)
find_package(GLUT)

IF (GLUT_FOUND)
   MESSAGE("GLUT found")
ENDIF (GLUT_FOUND)

IF (OPENGL_FOUND)
   MESSAGE("OpenGL found")
ENDIF (OPENGL_FOUND)

#add_subdirectory(utilities)
#add_subdirectory(random)

#ADD_DEFINITIONS(-g)

include_directories(
  ${glew_inc_path} 
  ../utilities 
  ../random
  ../graphics_utils
  ../timing
)

set (libraries 
	${graphics}
	${utilities}
	${random}
	${OPENGL_LIBRARIES}
	${GLUT_LIBRARIES}
	${glew_lib}
	${timing}
)

add_executable(pingpong ${src})
target_link_libraries(pingpong ${libraries})

