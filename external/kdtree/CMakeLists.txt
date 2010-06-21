cmake_minimum_required(VERSION 2.6)

project(kdtree)
set(SOURCES KDTree.h )

include_directories( . )
add_library(${PROJECT_NAME} STATIC ${SOURCES})
