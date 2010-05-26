PROJECT (helloworld)
CMAKE_MINIMUM_REQUIRED (VERSION 2.6)

#ENABLE_TESTING()

SET (TEST_NAME ${PROJECT_NAME}.x) 

SET (ARGV1 0)

# NOTE: We dont put an extension on the source file. CMake will take one
# 	source file by that name at runtime. Careful with this; we should 
# 	always name our sources
SET (PROJECT_SOURCES 
	main
)


ADD_EXECUTABLE (${TEST_NAME} ${PROJECT_SOURCES})
#target_link_libraries(${PROJECT_NAME} ${libname})

ADD_TEST (${PROJECT_NAME}_test1 ${EXECUTABLE_OUTPUT_PATH}/${TEST_NAME})
#ADD_TEST (${PROJECT_NAME}_test2 ${EXECUTABLE_OUTPUT_PATH}/${TEST_NAME} "${ARGV1}")



