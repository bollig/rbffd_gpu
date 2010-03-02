MESSAGE(STATUS "LINUX detected.")

# One or more dirs split by spaces. This is a command so it can be called multiple times
INCLUDE_DIRECTORIES (AFTER
)

# One or more dirs split by spaces. This is a command so it can be called multiple times
LINK_DIRECTORIES (AFTER
)


# Additional libraries required by this OS
# NOTE: order of libraries is important in Linux. 
# 	does not matter on macOSX
SET (ADDITIONAL_REQUIRED_LIBRARIES 
)

