
project(timers)
cmake_minimum_required(VERSION 2.6)
set(timings timer_eb.cpp)


include_directories( . )
add_library(${PROJECT_NAME} STATIC ${timings})

add_executable(eexec test_timer.cpp)
target_link_libraries(eexec ${PROJECT_NAME})
