cmake_minimum_required(VERSION 2.6)

project(settings CXX)
set(SOURCES 
    projectsettings.cpp
)
include_directories( . ${timers_SOURCE_DIR})
add_library(${PROJECT_NAME} STATIC ${SOURCES})
