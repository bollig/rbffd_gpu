FIND_LIBRARY (accelerate NAMES Accelerate)
MESSAGE(STATUS "MacOS X detected. Added '-framework Accelerate' to compiler flags")

# One or more dirs split by spaces. This is a command so it can be called multiple times
INCLUDE_DIRECTORIES (
	/sw/include
)

# One or more dirs split by spaces. This is a command so it can be called multiple times
LINK_DIRECTORIES (
	/sw/lib
)


# Additional libraries required by this OS
# NOTE: order of libraries is important in Linux. 
# 	does not matter on macOSX
SET (ADDITIONAL_REQUIRED_LIBRARIES 
	${accelerate}
)

