# Modified 5/4/09 Evan Bollig
# Added directories to search for Doxyfile.in so I can have 3rd party modules in a CMake subdir
# of the current source directory
# NOTE: UseDoxygen is based on code from http://tobias.rautenkranz.ch/cmake/doxygen/
# Addes options to enable/disable Latex. Also addes 

# - Run Doxygen
#
# Adds a doxygen target that runs doxygen to generate the html
# and optionally the LaTeX API documentation.
# The doxygen target is added to the doc target as dependency.
# i.e.: the API documentation is built with:
#  make doc
#
# USAGE: GLOBAL INSTALL
#
# Install it with:
#  cmake ./ && sudo make install
# Add the following to the CMakeLists.txt of your project:
#  include(UseDoxygen OPTIONAL)
# Optionally copy Doxyfile.in in the directory of CMakeLists.txt and edit it.
#
# USAGE: INCLUDE IN PROJECT
#
#  set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
#  include(UseDoxygen)
# Add the Doxyfile.in and UseDoxygen.cmake files to the projects source directory.
#
#
# Variables you may define are:
#  DOXYFILE_OUTPUT_DIR - Path where the Doxygen output is stored. Defaults to "doc".
#
#  DOXYFILE_LATEX_DIR - Directory where the Doxygen LaTeX output is stored. Defaults to "latex".
#
#  DOXYFILE_HTML_DIR - Directory where the Doxygen html output is stored. Defaults to "html".
#

#
#  Copyright (c) 2009 Tobias Rautenkranz <tobias@rautenkranz.ch>
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

macro(usedoxygen_set_default name value)
	if(NOT DEFINED "${name}")
		set("${name}" "${value}")
	endif()
endmacro()

find_package(Doxygen)

if(DOXYGEN_FOUND)
	find_file(DOXYFILE_IN "Doxyfile.in"
			PATHS "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/CMake" "${CMAKE_CURRENT_SOURCE_DIR}/CMake/UseDoxygen" "${CMAKE_ROOT}/Modules/" ${CMAKE_MODULE_PATH} "${USE_DOXYGEN_PATH}")

	include(FindPackageHandleStandardArgs)
	find_package_handle_standard_args(Doxyfile.in DEFAULT_MSG DOXYFILE_IN)
endif()

if(DOXYGEN_FOUND AND DOXYFILE_IN)

	#added by Evan Bollig 1/26/10

	if (UNIX)
		MESSAGE (STATUS "Doxygen on Unix; Loading SubVersion Filter")
		find_file(SUBVERSION_FILTER "subversionfilter.sh"
			PATHS "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/CMake" "${CMAKE_CURRENT_SOURCE_DIR}/CMake/UseDoxygen" "${CMAKE_ROOT}/Modules/" ${CMAKE_MODULE_PATH} "${USE_DOXYGEN_PATH}")

		set (UNIX_SVN_FILTER "/bin/sh ${SUBVERSION_FILTER}")
	endif (UNIX)

	add_custom_target(doxygen ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

	usedoxygen_set_default(DOXYFILE_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/doc")
	usedoxygen_set_default(DOXYFILE_HTML_DIR "html")

	set_property(DIRECTORY APPEND PROPERTY
			ADDITIONAL_MAKE_CLEAN_FILES "${DOXYFILE_OUTPUT_DIR}/${DOXYFILE_HTML_DIR}")

	set(DOXYFILE_LATEX "NO")
	set(DOXYFILE_PDFLATEX "NO")
	set(DOXYFILE_DOT "NO")

	find_package(LATEX)
	option(DOXYGEN_OUTPUT_LATEX "Should Doxygen output LaTeX in addition to HTML?" OFF)
	if(LATEX_COMPILER AND MAKEINDEX_COMPILER AND DOXYGEN_OUTPUT_LATEX)
		set(DOXYFILE_LATEX "YES")
		usedoxygen_set_default(DOXYFILE_LATEX_DIR "latex")

		set_property(DIRECTORY APPEND PROPERTY
				ADDITIONAL_MAKE_CLEAN_FILES
				"${DOXYFILE_OUTPUT_DIR}/${DOXYFILE_LATEX_DIR}")

		if(PDFLATEX_COMPILER)
			set(DOXYFILE_PDFLATEX "YES")
		endif()
		if(DOXYGEN_DOT_EXECUTABLE)
			set(DOXYFILE_DOT "YES")
		endif()

		add_custom_command(TARGET doxygen
			POST_BUILD
			COMMAND ${CMAKE_MAKE_PROGRAM}
			WORKING_DIRECTORY "${DOXYFILE_OUTPUT_DIR}/${DOXYFILE_LATEX_DIR}")
	endif()


	configure_file(${DOXYFILE_IN} Doxyfile ESCAPE_QUOTES IMMEDIATE @ONLY)

	get_target_property(DOC_TARGET doc TYPE)
	if(NOT DOC_TARGET)
		add_custom_target(doc)
	endif()
		
	add_dependencies(doc doxygen)
endif()
