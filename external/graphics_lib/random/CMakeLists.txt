
project(random)
set(random newran.cpp myexcept.cpp extreal.cpp hist.cpp)

include_directories( . )
add_library(random STATIC ${random})
