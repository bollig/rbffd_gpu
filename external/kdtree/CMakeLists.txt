cmake_minimum_required(VERSION 2.6)

project(kdtree CXX)
set(SOURCES 
	kdtree_ball_query.cpp
        kdtree_delete.cpp
        kdtree_nearest_neighbor.cpp
	kdtree_build.cpp
	kdtree_k_nearest_neighbors.cpp  
	kdtree_range_query.cpp
)
#KDTree.h
include_directories( . ${timers_SOURCE_DIR})
add_definitions( -DCPPONLY )
add_library(${PROJECT_NAME} STATIC ${SOURCES})
