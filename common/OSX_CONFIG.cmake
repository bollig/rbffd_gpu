FIND_LIBRARY (accelerate NAMES Accelerate)
# CLAPACK is available from netlib.org or linux repositories
# Use Accelerate library on OSX
FIND_LIBRARY (clapack clapack PATHS /usr/lib) 


MESSAGE(STATUS "MacOS X detected. Added '-framework Accelerate' to compiler flags")

# One or more dirs split by spaces. This is a command so it can be called multiple times
INCLUDE_DIRECTORIES (AFTER
	/sw/include
)

# One or more dirs split by spaces. This is a command so it can be called multiple times
LINK_DIRECTORIES (AFTER
	/sw/lib
)
	
# Additional libraries required by this OS
# NOTE: order of libraries is important in Linux. 
# 	does not matter on macOSX
SET (ADDITIONAL_REQUIRED_LIBRARIES 
	${clapack}
	${accelerate}
)

