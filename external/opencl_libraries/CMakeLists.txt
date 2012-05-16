
###############################################
# External includes we need on the include path
###############################################

OPTION (USE_CL11 "Enable/Disable the use of the OpenCL 1.1 spec (Default: ON)" ON)

MESSAGE(STATUS "USE_CL11 = ${USE_CL11}")

# NOTE: we need the ; here or cmake will not interpret the line properly
SET (ENV{EXTERNAL_INCLUDE_DIRS} 
    "$ENV{EXTERNAL_INCLUDE_DIRS};${CMAKE_CURRENT_SOURCE_DIR}/viennacl-1.3.0/"
    )


MESSAGE(STATUS "EXTERNAL OpenCL LIBRARIES ADDED TO VAR: ENV{EXTERNAL_INCLUDE_DIRS}")

# We can switch to opencl 1.1 if we have a boolean set to true: 
IF ( USE_CL11 )
    # NOTE: we need the ; here or cmake will not interpret the line properly
    SET (ENV{EXTERNAL_INCLUDE_DIRS} 
        "$ENV{EXTERNAL_INCLUDE_DIRS};${CMAKE_CURRENT_SOURCE_DIR}/opencl11"
        )
    MESSAGE(STATUS "OpenCL 1.1 HEADERS ADDED TO VAR: ENV{EXTERNAL_INCLUDE_DIRS}")
ELSE ( USE_CL11 )
    SET (ENV{EXTERNAL_INCLUDE_DIRS} 
        "$ENV{EXTERNAL_INCLUDE_DIRS};${CMAKE_CURRENT_SOURCE_DIR}/opencl10"
        )
    MESSAGE(STATUS "OpenCL 1.0 HEADERS ADDED TO VAR: ENV{EXTERNAL_INCLUDE_DIRS}")
ENDIF (USE_CL11)


